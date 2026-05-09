//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleanerCheck.h"
#include "../ClangTidyCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "../ClangTidyOptions.h"
#include "../utils/OptionsUtils.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/IncludeSpeller.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

static bool matchesAnyRegex(llvm::ArrayRef<llvm::Regex> Regexes,
                            llvm::StringRef Path) {
  return llvm::any_of(Regexes,
                      [&](const llvm::Regex &R) { return R.match(Path); });
}

IncludeCleanerCheck::IncludeCleanerCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreHeaders(
          utils::options::parseStringList(Options.get("IgnoreHeaders", ""))),
      FragmentHeaderPatterns(
          utils::options::parseStringList(Options.get("FragmentHeaders", ""))),
      FragmentDependencyCommentFormat(
          Options.get("FragmentDependencyCommentFormat", "")),
      DeduplicateFindings(Options.get("DeduplicateFindings", true)),
      UnusedIncludes(Options.get("UnusedIncludes", true)),
      MissingIncludes(Options.get("MissingIncludes", true)) {
  for (const auto &Header : IgnoreHeaders) {
    if (!llvm::Regex{Header}.isValid())
      configurationDiag("Invalid ignore headers regex '%0'") << Header;
    std::string HeaderSuffix{Header.str()};
    if (!Header.ends_with('$'))
      HeaderSuffix += '$';
    IgnoreHeadersRegex.emplace_back(HeaderSuffix);
  }
  for (const StringRef Pattern : FragmentHeaderPatterns) {
    llvm::Regex CompiledRegex(Pattern);
    std::string RegexError;
    if (!CompiledRegex.isValid(RegexError)) {
      configurationDiag("Invalid fragment headers regular expression '%0': %1")
          << Pattern << RegexError;
      continue;
    }
    FragmentHeaderRegexes.push_back(std::move(CompiledRegex));
  }

  if (UnusedIncludes == false && MissingIncludes == false)
    this->configurationDiag("The check 'misc-include-cleaner' will not "
                            "perform any analysis because 'UnusedIncludes' and "
                            "'MissingIncludes' are both false.");
}

void IncludeCleanerCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreHeaders",
                utils::options::serializeStringList(IgnoreHeaders));
  Options.store(Opts, "FragmentHeaders",
                utils::options::serializeStringList(FragmentHeaderPatterns));
  Options.store(Opts, "FragmentDependencyCommentFormat",
                FragmentDependencyCommentFormat);
  Options.store(Opts, "DeduplicateFindings", DeduplicateFindings);
  Options.store(Opts, "UnusedIncludes", UnusedIncludes);
  Options.store(Opts, "MissingIncludes", MissingIncludes);
}

bool IncludeCleanerCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return !LangOpts.ObjC;
}

void IncludeCleanerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl().bind("top"), this);
}

void IncludeCleanerCheck::registerPPCallbacks(const SourceManager &SM,
                                              Preprocessor *PP,
                                              Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(RecordedPreprocessor.record(*PP));
  this->PP = PP;
  RecordedPI.record(*PP);
}

bool IncludeCleanerCheck::shouldIgnore(const include_cleaner::Header &H) {
  return llvm::any_of(IgnoreHeadersRegex, [&H](const llvm::Regex &R) {
    switch (H.kind()) {
    case include_cleaner::Header::Standard:
      // We don't trim angle brackets around standard library headers
      // deliberately, so that they are only matched as <vector>, otherwise
      // having just `.*/vector` might yield false positives.
      return R.match(H.standard().name());
    case include_cleaner::Header::Verbatim:
      return R.match(H.verbatim().trim("<>\""));
    case include_cleaner::Header::Physical:
      return R.match(H.physical().getFileEntry().tryGetRealPathName());
    }
    llvm_unreachable("Unknown Header kind.");
  });
}

void IncludeCleanerCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceManager *SM = Result.SourceManager;
  const FileEntry *MainFile = SM->getFileEntryForID(SM->getMainFileID());
  llvm::SmallVector<Decl *> RootDecls;
  for (Decl *D : Result.Nodes.getNodeAs<TranslationUnitDecl>("top")->decls()) {
    // FIXME: Filter out implicit template specializations.
    RootDecls.push_back(D);
  }
  std::function<bool(llvm::StringRef)> FragmentHeaderFilter;
  if (!FragmentHeaderRegexes.empty()) {
    FragmentHeaderFilter = [&](llvm::StringRef Path) {
      return matchesAnyRegex(FragmentHeaderRegexes, Path);
    };
  }
  const include_cleaner::AnalysisOptions AnalyzeOptions{
      [&](const include_cleaner::Header &H) { return shouldIgnore(H); },
      FragmentHeaderFilter};
  const include_cleaner::AnalysisResults Results =
      analyze(RootDecls, RecordedPreprocessor.MacroReferences,
              RecordedPreprocessor.Includes, &RecordedPI, *PP, AnalyzeOptions);

  const StringRef Code = SM->getBufferData(SM->getMainFileID());
  auto FileStyle =
      format::getStyle(format::DefaultFormatStyle, getCurrentMainFile(),
                       format::DefaultFallbackStyle, Code,
                       &SM->getFileManager().getVirtualFileSystem());
  if (!FileStyle)
    FileStyle = format::getLLVMStyle();

  if (UnusedIncludes) {
    for (const auto *Inc : Results.Unused) {
      diag(Inc->HashLocation, "included header %0 is not used directly")
          << llvm::sys::path::filename(Inc->Spelled,
                                       llvm::sys::path::Style::posix)
          << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
                 SM->translateLineCol(SM->getMainFileID(), Inc->Line, 1),
                 SM->translateLineCol(SM->getMainFileID(), Inc->Line + 1, 1)));
    }
  }

  if (MissingIncludes) {
    const tooling::HeaderIncludes HeaderIncludes(getCurrentMainFile(), Code,
                                                 FileStyle->IncludeStyle);
    // Deduplicate insertions when running in bulk fix mode.
    llvm::StringSet<> InsertedHeaders{};
    llvm::DenseSet<include_cleaner::Symbol> SeenSymbols;
    auto DiagLocation = [&](const include_cleaner::MissingIncludeRef &Missing) {
      if (Missing.Origin == include_cleaner::SymbolReferenceOrigin::Fragment &&
          Missing.FragmentInclude) {
        // Fragment refs are reported on their direct include site in the main
        // file to preserve the main-file-only diagnostics contract.
        return Missing.FragmentInclude->HashLocation;
      }
      return SM->getSpellingLoc(Missing.Ref.RefLocation);
    };
    for (const auto &Missing : Results.MissingRefs) {
      // Process each symbol once to reduce noise in the findings.
      // Tidy checks are used in two different workflows:
      // - Ones that show all the findings for a given file. For such
      // workflows there is not much point in showing all the occurences,
      // as one is enough to indicate the issue.
      // - Ones that show only the findings on changed pieces. For such
      // workflows it's useful to show findings on every reference of a
      // symbol as otherwise tools might give incosistent results
      // depending on the parts of the file being edited. But it should
      // still help surface findings for "new violations" (i.e.
      // dependency did not exist in the code at all before).
      if (DeduplicateFindings && !SeenSymbols.insert(Missing.Ref.Target).second)
        continue;
      assert(!Missing.Providers.empty() && "missing include without provider");
      const std::string Spelling = include_cleaner::spellHeader(
          {Missing.Providers.front(), PP->getHeaderSearchInfo(), MainFile});
      const bool Angled = StringRef{Spelling}.starts_with('<');
      // We might suggest insertion of an existing include in edge cases, e.g.,
      // include is present in a PP-disabled region, or spelling of the header
      // turns out to be the same as one of the unresolved includes in the
      // main file.
      if (auto Replacement =
              HeaderIncludes.insert(StringRef{Spelling}.trim("\"<>"), Angled,
                                    tooling::IncludeDirective::Include)) {
        const DiagnosticBuilder DB =
            diag(DiagLocation(Missing),
                 "no header providing \"%0\" is directly included")
            << Missing.Ref.Target.name();
        if (areDiagsSelfContained() ||
            InsertedHeaders.insert(Replacement->getReplacementText()).second) {
          DB << FixItHint::CreateInsertion(
              SM->getComposedLoc(SM->getMainFileID(), Replacement->getOffset()),
              Replacement->getReplacementText());
        }
      }
    }
  }

  if (FragmentDependencyCommentFormat.empty())
    return;

  const include_cleaner::FixIncludesOptions FixOptions{
      FragmentDependencyCommentFormat};
  const include_cleaner::IncludeFixes Fixes = computeIncludeFixes(
      Results, getCurrentMainFile(), Code, *FileStyle, FixOptions);
  for (const auto &Comment : Fixes.FragmentComments) {
    if (Comment.Status ==
        include_cleaner::FragmentDependencyCommentStatus::AlreadyPresent) {
      continue;
    }
    std::string FragmentList;
    for (const auto *Fragment : Comment.Fragments) {
      if (!FragmentList.empty())
        FragmentList += ", ";
      FragmentList += Fragment->quote();
    }
    const DiagnosticBuilder DB =
        diag(Comment.Preserved->HashLocation,
             "included header %0 is used only by fragment header(s) %1")
        << llvm::sys::path::filename(Comment.Preserved->Spelled,
                                     llvm::sys::path::Style::posix)
        << FragmentList;
    if (const auto Replacement = Comment.Replacement) {
      DB << FixItHint::CreateInsertion(
          SM->getComposedLoc(SM->getMainFileID(), Replacement->getOffset()),
          Replacement->getReplacementText());
    }
  }
}

} // namespace clang::tidy::misc
