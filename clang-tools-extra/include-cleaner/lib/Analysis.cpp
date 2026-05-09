//===--- Analysis.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
#include "AnalysisInternal.h"
#include "TypesInternal.h"
#include "clang-include-cleaner/IncludeSpeller.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/DirectoryEntry.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <climits>
#include <optional>
#include <string>

namespace clang::include_cleaner {

namespace {

struct ClassifiedReference {
  SymbolReferenceOrigin Origin;
  // Null if the fragment file has multiple direct include sites.
  const Include *FragmentInclude = nullptr;
};

class FragmentTracker {
public:
  FragmentTracker(const Includes &Inc, const SourceManager &SM,
                  const std::function<bool(llvm::StringRef)> &HeaderFilter)
      : SM(SM) {
    if (!HeaderFilter)
      return;

    for (const Include &I : Inc.all()) {
      if (!I.Resolved ||
          locateInMainFile(I.HashLocation, SM) != MainFileLocation::MainFile) {
        continue;
      }

      // Match the canonical path first, but fall back to the spelled include
      // so generated paths can still be configured even when resolution loses
      // that spelling detail.
      const llvm::SmallString<128> ResolvedPath =
          normalizePath(I.Resolved->getName());
      bool IsFragment = HeaderFilter(ResolvedPath);
      if (!IsFragment) {
        const llvm::SmallString<128> SpelledPath = normalizePath(I.Spelled);
        if (!SpelledPath.empty())
          IsFragment = HeaderFilter(SpelledPath);
      }
      if (!IsFragment)
        continue;

      // Fragments are intentionally direct-includes-only for now. If a
      // fragment includes another file, that nested include keeps normal
      // header semantics.
      IncludeSitesByFile[&I.Resolved->getFileEntry()].push_back(&I);
      DirectIncludes.insert(&I);
    }
  }

  std::optional<ClassifiedReference> classify(FileID FID) const {
    if (FID == SM.getMainFileID())
      return ClassifiedReference{SymbolReferenceOrigin::MainFile};
    if (FID == SM.getPreambleFileID())
      return ClassifiedReference{SymbolReferenceOrigin::Preamble};
    auto FE = SM.getFileEntryRefForID(FID);
    if (!FE)
      return std::nullopt;
    auto It = IncludeSitesByFile.find(&FE->getFileEntry());
    if (It == IncludeSitesByFile.end())
      return std::nullopt;
    if (It->second.size() != 1)
      return ClassifiedReference{SymbolReferenceOrigin::Fragment};
    return ClassifiedReference{SymbolReferenceOrigin::Fragment,
                               It->second.front()};
  }

  bool isDirectInclude(const Include *I) const {
    return DirectIncludes.contains(I);
  }

private:
  const SourceManager &SM;
  llvm::DenseMap<const FileEntry *, llvm::SmallVector<const Include *>>
      IncludeSitesByFile;
  llvm::DenseSet<const Include *> DirectIncludes;
};

using UsedSymbolWithOriginCB = llvm::function_ref<void(
    const SymbolReference &Ref, llvm::ArrayRef<Header> Providers,
    const ClassifiedReference &Info)>;

bool shouldIgnoreMacroReference(const Preprocessor &PP, const Macro &M) {
  auto &MutablePP = const_cast<Preprocessor &>(PP);
  auto MD = MutablePP.getMacroDefinitionAtLoc(M.Name, M.Definition);
  auto *MI = MD ? MD.getMacroInfo() : PP.getMacroInfo(M.Name);
  // Macros that expand to themselves are confusing from user's point of view.
  // They usually aspect the usage to be attributed to the underlying decl and
  // not the macro definition. So ignore such macros (e.g. std{in,out,err} are
  // implementation defined macros, that just resolve to themselves in
  // practice).
  return MI && MI->getNumTokens() == 1 && MI->isObjectLike() &&
         MI->getReplacementToken(0).getIdentifierInfo() == M.Name;
}

void walkUsedWithOrigins(
    llvm::ArrayRef<Decl *> ASTRoots, llvm::ArrayRef<SymbolReference> MacroRefs,
    const PragmaIncludes *PI, const Preprocessor &PP, UsedSymbolWithOriginCB CB,
    llvm::function_ref<std::optional<ClassifiedReference>(FileID)> Classify) {
  const auto &SM = PP.getSourceManager();
  for (auto *Root : ASTRoots) {
    walkAST(*Root, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
      auto SpellLoc = SM.getSpellingLoc(Loc);
      // Tokens resulting from macro concatenation ends up in scratch space and
      // clang currently doesn't have a good/simple APIs for tracking where
      // pieces of a concataned token originated from.
      // So we use the macro expansion location instead, and downgrade reference
      // type to ambigious to prevent false negatives.
      if (SM.isWrittenInScratchSpace(SpellLoc)) {
        Loc = SM.getExpansionLoc(Loc);
        if (RT == RefType::Explicit)
          RT = RefType::Ambiguous;
        SpellLoc = SM.getSpellingLoc(Loc);
      }
      auto FID = SM.getFileID(SpellLoc);
      auto Info = Classify(FID);
      if (!Info)
        return;
      // FIXME: Most of the work done here is repetitive. It might be useful to
      // have a cache/batching.
      SymbolReference SymRef{ND, Loc, RT};
      return CB(SymRef, headersForSymbol(ND, PP, PI), *Info);
    });
  }

  for (const SymbolReference &MacroRef : MacroRefs) {
    assert(MacroRef.Target.kind() == Symbol::Macro);
    if (shouldIgnoreMacroReference(PP, MacroRef.Target.macro()))
      continue;
    auto FID = SM.getFileID(SM.getSpellingLoc(MacroRef.RefLocation));
    auto Info = Classify(FID);
    if (!Info)
      continue;
    CB(MacroRef, headersForSymbol(MacroRef.Target, PP, PI), *Info);
  }
}

bool isFilteredInclude(const Include &I,
                       llvm::function_ref<bool(const Header &)> HeaderFilter,
                       const Preprocessor &PP) {
  if (I.Angled) {
    auto Lang = PP.getLangOpts().CPlusPlus ? tooling::stdlib::Lang::CXX
                                           : tooling::stdlib::Lang::C;
    if (auto StdHeader = tooling::stdlib::Header::named(I.quote(), Lang);
        StdHeader && HeaderFilter(*StdHeader)) {
      return true;
    }
  }
  return I.Resolved && HeaderFilter(*I.Resolved);
}

bool shouldSuppressIncludeDiagnostic(
    const Include &I, FileEntryRef MainFile,
    llvm::function_ref<bool(const Header &)> HeaderFilter,
    const PragmaIncludes *PI, const Preprocessor &PP,
    OptionalDirectoryEntryRef ResourceDir) {
  if (!I.Resolved || I.Resolved->getDir() == ResourceDir ||
      isFilteredInclude(I, HeaderFilter, PP)) {
    return true;
  }
  if (!PI)
    return false;
  if (PI->shouldKeep(*I.Resolved))
    return true;
  // Check if main file is the public interface for a private header. If so
  // we shouldn't diagnose it as unused or record fragment dependencies.
  if (auto PHeader = PI->getPublic(*I.Resolved); !PHeader.empty()) {
    PHeader = PHeader.trim("<>\"");
    if (MainFile.getName().ends_with(PHeader))
      return true;
  }
  return false;
}

void walkUsedInFiles(llvm::ArrayRef<Decl *> ASTRoots,
                     llvm::ArrayRef<SymbolReference> MacroRefs,
                     const PragmaIncludes *PI, const Preprocessor &PP,
                     UsedSymbolCB CB,
                     llvm::function_ref<bool(FileID)> IsMainFile) {
  walkUsedWithOrigins(
      ASTRoots, MacroRefs, PI, PP,
      [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers,
          const ClassifiedReference &) { CB(Ref, Providers); },
      [&](FileID FID) -> std::optional<ClassifiedReference> {
        if (!IsMainFile(FID))
          return std::nullopt;
        return ClassifiedReference{SymbolReferenceOrigin::MainFile};
      });
}

} // namespace

void walkUsed(llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs,
              const PragmaIncludes *PI, const Preprocessor &PP,
              UsedSymbolCB CB) {
  const auto &SM = PP.getSourceManager();
  walkUsedInFiles(ASTRoots, MacroRefs, PI, PP, CB, [&](FileID FID) {
    return FID == SM.getMainFileID() || FID == SM.getPreambleFileID();
  });
}

namespace {

class IncludeUsage {
public:
  void mark(const Include *I, const ClassifiedReference &Info) {
    Used.insert(I);
    if (Info.Origin != SymbolReferenceOrigin::Fragment) {
      UsedByMainOrPreamble.insert(I);
      return;
    }

    UsedByFragment.insert(I);
    if (!Info.FragmentInclude)
      return;
    auto &Reasons = ByPreserved[I];
    if (!llvm::is_contained(Reasons, Info.FragmentInclude))
      Reasons.push_back(Info.FragmentInclude);
  }

  bool contains(const Include *I) const { return Used.contains(I); }

  bool isFragmentOnly(const Include *I) const {
    return UsedByFragment.contains(I) && !UsedByMainOrPreamble.contains(I);
  }

  llvm::SmallVector<const Include *> fragmentSites(const Include *I) const {
    auto It = ByPreserved.find(I);
    if (It == ByPreserved.end())
      return {};
    return It->second;
  }

private:
  llvm::DenseSet<const Include *> Used;
  llvm::DenseSet<const Include *> UsedByMainOrPreamble;
  llvm::DenseSet<const Include *> UsedByFragment;
  llvm::DenseMap<const Include *, llvm::SmallVector<const Include *>>
      ByPreserved;
};

class MissingIncludeCollector {
public:
  void add(llvm::StringRef Spelling, const Header &Provider) {
    Missing.try_emplace(Spelling, Provider);
  }

  std::vector<MissingInclude> take() && {
    std::vector<MissingInclude> Result;
    Result.reserve(Missing.size());
    for (auto &E : Missing)
      Result.push_back(MissingInclude{E.first().str(), E.second});
    llvm::sort(Result, [](const MissingInclude &L, const MissingInclude &R) {
      return L.Spelling < R.Spelling;
    });
    return Result;
  }

private:
  llvm::StringMap<Header> Missing;
};

} // namespace

AnalysisResults analyze(llvm::ArrayRef<Decl *> ASTRoots,
                        llvm::ArrayRef<SymbolReference> MacroRefs,
                        const Includes &Inc, const PragmaIncludes *PI,
                        const Preprocessor &PP,
                        const AnalysisOptions &Options) {
  auto &SM = PP.getSourceManager();
  const auto MainFile = *SM.getFileEntryRefForID(SM.getMainFileID());
  auto HeaderFilter = [&](const Header &H) {
    return Options.HeaderFilter && Options.HeaderFilter(H);
  };
  FragmentTracker Fragments(Inc, SM, Options.FragmentHeaderFilter);
  IncludeUsage Usage;
  MissingIncludeCollector MissingIncludes;
  std::vector<MissingIncludeRef> MissingRefs;
  OptionalDirectoryEntryRef ResourceDir =
      PP.getHeaderSearchInfo().getModuleMap().getBuiltinDir();

  walkUsedWithOrigins(
      ASTRoots, MacroRefs, PI, PP,
      [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers,
          const ClassifiedReference &Info) {
        bool Satisfied = false;
        for (const Header &H : Providers) {
          if (H.kind() == Header::Physical &&
              (H.physical() == MainFile ||
               H.physical().getDir() == ResourceDir)) {
            Satisfied = true;
          }
          for (const Include *I : Inc.match(H)) {
            Usage.mark(I, Info);
            Satisfied = true;
          }
        }

        if (Satisfied || Providers.empty() || Ref.RT != RefType::Explicit)
          return;
        if (HeaderFilter(Providers.front()))
          return;

        auto Spelling = spellHeader(
            {Providers.front(), PP.getHeaderSearchInfo(), MainFile});
        for (const Include *I : Inc.match(Header{Spelling})) {
          Usage.mark(I, Info);
          Satisfied = true;
        }
        if (Satisfied)
          return;

        // MissingIncludes drives edits, while MissingRefs preserves where the
        // unsatisfied use came from for higher-level diagnostics.
        MissingIncludes.add(Spelling, Providers.front());
        MissingRefs.push_back(
            MissingIncludeRef{Ref, llvm::SmallVector<Header>(Providers),
                              Info.Origin, Info.FragmentInclude});
      },
      [&](FileID FID) { return Fragments.classify(FID); });

  AnalysisResults Results{{}, {}, std::move(MissingRefs), {}};
  for (const Include &I : Inc.all()) {
    bool Suppressed = shouldSuppressIncludeDiagnostic(I, MainFile, HeaderFilter,
                                                      PI, PP, ResourceDir);
    if (Usage.contains(&I)) {
      const llvm::SmallVector<const Include *> FragmentSites =
          Usage.fragmentSites(&I);
      // Ambiguous fragment sites still preserve the header, but get no comment.
      if (!Suppressed && Usage.isFragmentOnly(&I) &&
          !Fragments.isDirectInclude(&I) && !FragmentSites.empty()) {
        Results.FragmentDependencies.push_back(
            FragmentDependency{&I, FragmentSites});
      }
      continue;
    }
    if (Suppressed)
      continue;
    Results.Unused.push_back(&I);
  }
  Results.MissingIncludes = std::move(MissingIncludes).take();
  return Results;
}

namespace {

// The fix planner only receives FileName and Code, so comment edits map
// Include::Line back to offsets in that text.
class LineIndex {
public:
  explicit LineIndex(llvm::StringRef Code) : Code(Code) {
    Starts.push_back(0);
    for (unsigned I = 0; I < Code.size(); ++I) {
      if (Code[I] == '\n')
        Starts.push_back(I + 1);
    }
    Starts.push_back(Code.size());
  }

  llvm::StringRef line(unsigned OneBasedLine) const {
    auto [Start, End] = bounds(OneBasedLine);
    return Code.slice(Start, End);
  }

  unsigned trimmedLineEnd(unsigned OneBasedLine) const {
    auto [Start, End] = bounds(OneBasedLine);
    while (End > Start && (Code[End - 1] == ' ' || Code[End - 1] == '\t'))
      --End;
    return End;
  }

private:
  std::pair<unsigned, unsigned> bounds(unsigned OneBasedLine) const {
    if (OneBasedLine == 0 || OneBasedLine >= Starts.size())
      return {Code.size(), Code.size()};
    unsigned Start = Starts[OneBasedLine - 1];
    unsigned End =
        Starts[OneBasedLine] == 0 ? Code.size() : Starts[OneBasedLine] - 1;
    if (End > Start && Code[End - 1] == '\r')
      --End;
    return {Start, End};
  }

  llvm::StringRef Code;
  llvm::SmallVector<unsigned> Starts;
};

std::string formatFragmentDependencyComment(
    llvm::StringRef Format, llvm::ArrayRef<const Include *> FragmentIncludes) {
  if (Format.empty())
    return {};

  std::string FragmentList;
  for (const Include *Fragment : FragmentIncludes) {
    if (!FragmentList.empty())
      FragmentList += ", ";
    FragmentList += Fragment->quote();
  }

  std::string Result;
  llvm::StringRef Remaining = Format;
  while (true) {
    auto Pos = Remaining.find("{0}");
    if (Pos == llvm::StringRef::npos) {
      Result += Remaining.str();
      return Result;
    }
    Result += Remaining.take_front(Pos).str();
    Result += FragmentList;
    Remaining = Remaining.drop_front(Pos + 3);
  }
}

FragmentDependencyComment inspectFragmentDependencyComment(
    const FragmentDependency &Dependency, llvm::StringRef FileName,
    const LineIndex &Lines, llvm::StringRef CommentFormat) {
  FragmentDependencyComment Comment{
      Dependency.Preserved, Dependency.Fragments,
      formatFragmentDependencyComment(CommentFormat, Dependency.Fragments),
      FragmentDependencyCommentStatus::CanInsert, std::nullopt};
  if (Comment.Text.empty())
    return Comment;

  llvm::StringRef Line = Lines.line(Dependency.Preserved->Line);
  size_t IncludePos = Line.find(Dependency.Preserved->quote());
  if (IncludePos == llvm::StringRef::npos) {
    Comment.Status = FragmentDependencyCommentStatus::ConflictingComment;
    return Comment;
  }

  llvm::StringRef Tail =
      Line.drop_front(IncludePos + Dependency.Preserved->quote().size());
  llvm::StringRef TrimmedTail = Tail.ltrim(" \t");
  if (TrimmedTail.empty()) {
    Comment.Status = FragmentDependencyCommentStatus::CanInsert;
    Comment.Replacement = tooling::Replacement(
        FileName, Lines.trimmedLineEnd(Dependency.Preserved->Line), 0,
        " // " + Comment.Text);
    return Comment;
  }

  if (TrimmedTail.starts_with("//")) {
    llvm::StringRef Existing = TrimmedTail.drop_front(2).trim();
    Comment.Status = Existing == Comment.Text
                         ? FragmentDependencyCommentStatus::AlreadyPresent
                         : FragmentDependencyCommentStatus::ConflictingComment;
    return Comment;
  }

  Comment.Status = FragmentDependencyCommentStatus::ConflictingComment;
  return Comment;
}

} // namespace

IncludeFixes computeIncludeFixes(const AnalysisResults &Results,
                                 llvm::StringRef FileName, llvm::StringRef Code,
                                 const format::FormatStyle &Style,
                                 const FixIncludesOptions &Options) {
  assert(Style.isCpp() && "Only C++ style supports include insertions!");
  IncludeFixes Fixes;
  tooling::Replacements &R = Fixes.Replacements;
  // Encode insertions/deletions in the magic way clang-format understands.
  for (const Include *I : Results.Unused)
    cantFail(R.add(tooling::Replacement(FileName, UINT_MAX, 1, I->quote())));
  for (const MissingInclude &Missing : Results.MissingIncludes)
    cantFail(R.add(tooling::Replacement(FileName, UINT_MAX, 0,
                                        "#include " + Missing.Spelling)));

  if (Options.FragmentDependencyCommentFormat.empty())
    return Fixes;

  LineIndex Lines(Code);
  for (const FragmentDependency &Dependency : Results.FragmentDependencies) {
    FragmentDependencyComment Comment = inspectFragmentDependencyComment(
        Dependency, FileName, Lines, Options.FragmentDependencyCommentFormat);
    if (Comment.Replacement)
      cantFail(R.add(*Comment.Replacement));
    Fixes.FragmentComments.push_back(std::move(Comment));
  }
  return Fixes;
}

std::string fixIncludes(const AnalysisResults &Results,
                        llvm::StringRef FileName, llvm::StringRef Code,
                        const format::FormatStyle &Style) {
  return fixIncludes(Results, FileName, Code, Style, {});
}

std::string fixIncludes(const AnalysisResults &Results,
                        llvm::StringRef FileName, llvm::StringRef Code,
                        const format::FormatStyle &Style,
                        const FixIncludesOptions &Options) {
  IncludeFixes Fixes =
      computeIncludeFixes(Results, FileName, Code, Style, Options);
  // "cleanup" actually turns the UINT_MAX replacements into concrete edits.
  auto Positioned = cantFail(
      format::cleanupAroundReplacements(Code, Fixes.Replacements, Style));
  return cantFail(tooling::applyAllReplacements(Code, Positioned));
}

} // namespace clang::include_cleaner
