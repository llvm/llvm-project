//===--- IncludeCleaner.cpp - Unused/Missing Headers Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleaner.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/IncludeSpeller.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <cassert>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clang::clangd {
namespace {

bool isIgnored(llvm::StringRef HeaderPath, HeaderFilter IgnoreHeaders) {
  // Convert the path to Unix slashes and try to match against the filter.
  llvm::SmallString<64> NormalizedPath(HeaderPath);
  llvm::sys::path::native(NormalizedPath, llvm::sys::path::Style::posix);
  for (auto &Filter : IgnoreHeaders) {
    if (Filter(NormalizedPath))
      return true;
  }
  return false;
}

bool mayConsiderUnused(const Inclusion &Inc, ParsedAST &AST,
                       const include_cleaner::PragmaIncludes *PI,
                       bool AnalyzeAngledIncludes) {
  assert(Inc.HeaderID);
  auto HID = static_cast<IncludeStructure::HeaderID>(*Inc.HeaderID);
  auto FE = AST.getSourceManager().getFileManager().getFileRef(
      AST.getIncludeStructure().getRealPath(HID));
  assert(FE);
  if (FE->getDir() == AST.getPreprocessor()
                          .getHeaderSearchInfo()
                          .getModuleMap()
                          .getBuiltinDir())
    return false;
  if (PI && PI->shouldKeep(*FE))
    return false;
  // FIXME(kirillbobyrev): We currently do not support the umbrella headers.
  // System headers are likely to be standard library headers.
  // Until we have good support for umbrella headers, don't warn about them
  // (unless analysis is explicitly enabled).
  if (Inc.Written.front() == '<') {
    if (tooling::stdlib::Header::named(Inc.Written))
      return true;
    if (!AnalyzeAngledIncludes)
      return false;
  }
  if (PI) {
    // Check if main file is the public interface for a private header. If so we
    // shouldn't diagnose it as unused.
    if (auto PHeader = PI->getPublic(*FE); !PHeader.empty()) {
      PHeader = PHeader.trim("<>\"");
      // Since most private -> public mappings happen in a verbatim way, we
      // check textually here. This might go wrong in presence of symlinks or
      // header mappings. But that's not different than rest of the places.
      if (AST.tuPath().ends_with(PHeader))
        return false;
    }
  }
  // Headers without include guards have side effects and are not
  // self-contained, skip them.
  if (!AST.getPreprocessor().getHeaderSearchInfo().isFileMultipleIncludeGuarded(
          *FE)) {
    dlog("{0} doesn't have header guard and will not be considered unused",
         FE->getName());
    return false;
  }
  return true;
}

std::vector<Diag> generateMissingIncludeDiagnostics(
    ParsedAST &AST, llvm::ArrayRef<MissingIncludeDiagInfo> MissingIncludes,
    llvm::StringRef Code, HeaderFilter IgnoreHeaders, const ThreadsafeFS &TFS) {
  std::vector<Diag> Result;
  const SourceManager &SM = AST.getSourceManager();
  const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID());

  auto FileStyle = getFormatStyleForFile(AST.tuPath(), Code, TFS, false);

  tooling::HeaderIncludes HeaderIncludes(AST.tuPath(), Code,
                                         FileStyle.IncludeStyle);
  for (const auto &SymbolWithMissingInclude : MissingIncludes) {
    llvm::StringRef ResolvedPath =
        SymbolWithMissingInclude.Providers.front().resolvedPath();
    if (isIgnored(ResolvedPath, IgnoreHeaders)) {
      dlog("IncludeCleaner: not diagnosing missing include {0}, filtered by "
           "config",
           ResolvedPath);
      continue;
    }

    std::string Spelling = include_cleaner::spellHeader(
        {SymbolWithMissingInclude.Providers.front(),
         AST.getPreprocessor().getHeaderSearchInfo(), MainFile});

    llvm::StringRef HeaderRef{Spelling};
    bool Angled = HeaderRef.starts_with("<");
    // We might suggest insertion of an existing include in edge cases, e.g.,
    // include is present in a PP-disabled region, or spelling of the header
    // turns out to be the same as one of the unresolved includes in the
    // main file.
    std::optional<tooling::Replacement> Replacement = HeaderIncludes.insert(
        HeaderRef.trim("\"<>"), Angled, tooling::IncludeDirective::Include);
    if (!Replacement.has_value())
      continue;

    Diag &D = Result.emplace_back();
    D.Message =
        llvm::formatv("No header providing \"{0}\" is directly included",
                      SymbolWithMissingInclude.Symbol.name());
    D.Name = "missing-includes";
    D.Source = Diag::DiagSource::Clangd;
    D.File = AST.tuPath();
    D.InsideMainFile = true;
    // We avoid the "warning" severity here in favor of LSP's "information".
    //
    // Users treat most warnings on code being edited as high-priority.
    // They don't think of include cleanups the same way: they want to edit
    // lines with existing violations without fixing them.
    // Diagnostics at the same level tend to be visually indistinguishable,
    // and a few missing includes can cause many diagnostics.
    // Marking these as "information" leaves them visible, but less intrusive.
    //
    // (These concerns don't apply to unused #include warnings: these are fewer,
    // they appear on infrequently-edited lines with few other warnings, and
    // the 'Unneccesary' tag often result in a different rendering)
    //
    // Usually clang's "note" severity usually has special semantics, being
    // translated into LSP RelatedInformation of a parent diagnostic.
    // But not here: these aren't processed by clangd's DiagnosticConsumer.
    D.Severity = DiagnosticsEngine::Note;
    D.Range = clangd::Range{
        offsetToPosition(Code,
                         SymbolWithMissingInclude.SymRefRange.beginOffset()),
        offsetToPosition(Code,
                         SymbolWithMissingInclude.SymRefRange.endOffset())};
    auto &F = D.Fixes.emplace_back();
    F.Message = "#include " + Spelling;
    TextEdit Edit = replacementToEdit(Code, *Replacement);
    F.Edits.emplace_back(std::move(Edit));
  }
  return Result;
}

std::vector<Diag> generateUnusedIncludeDiagnostics(
    PathRef FileName, llvm::ArrayRef<const Inclusion *> UnusedIncludes,
    llvm::StringRef Code, HeaderFilter IgnoreHeaders) {
  std::vector<Diag> Result;
  for (const auto *Inc : UnusedIncludes) {
    if (isIgnored(Inc->Resolved, IgnoreHeaders))
      continue;
    Diag &D = Result.emplace_back();
    D.Message =
        llvm::formatv("included header {0} is not used directly",
                      llvm::sys::path::filename(
                          Inc->Written.substr(1, Inc->Written.size() - 2),
                          llvm::sys::path::Style::posix));
    D.Name = "unused-includes";
    D.Source = Diag::DiagSource::Clangd;
    D.File = FileName;
    D.InsideMainFile = true;
    D.Severity = DiagnosticsEngine::Warning;
    D.Tags.push_back(Unnecessary);
    D.Range = rangeTillEOL(Code, Inc->HashOffset);
    // FIXME(kirillbobyrev): Removing inclusion might break the code if the
    // used headers are only reachable transitively through this one. Suggest
    // including them directly instead.
    // FIXME(kirillbobyrev): Add fix suggestion for adding IWYU pragmas
    // (keep/export) remove the warning once we support IWYU pragmas.
    auto &F = D.Fixes.emplace_back();
    F.Message = "remove #include directive";
    F.Edits.emplace_back();
    F.Edits.back().range.start.line = Inc->HashLine;
    F.Edits.back().range.end.line = Inc->HashLine + 1;
  }
  return Result;
}

std::optional<Fix>
removeAllUnusedIncludes(llvm::ArrayRef<Diag> UnusedIncludes) {
  if (UnusedIncludes.empty())
    return std::nullopt;

  Fix RemoveAll;
  RemoveAll.Message = "remove all unused includes";
  for (const auto &Diag : UnusedIncludes) {
    assert(Diag.Fixes.size() == 1 && "Expected exactly one fix.");
    RemoveAll.Edits.insert(RemoveAll.Edits.end(),
                           Diag.Fixes.front().Edits.begin(),
                           Diag.Fixes.front().Edits.end());
  }
  return RemoveAll;
}

std::optional<Fix>
addAllMissingIncludes(llvm::ArrayRef<Diag> MissingIncludeDiags) {
  if (MissingIncludeDiags.empty())
    return std::nullopt;

  Fix AddAllMissing;
  AddAllMissing.Message = "add all missing includes";
  // A map to deduplicate the edits with the same new text.
  // newText (#include "my_missing_header.h") -> TextEdit.
  std::map<std::string, TextEdit> Edits;
  for (const auto &Diag : MissingIncludeDiags) {
    assert(Diag.Fixes.size() == 1 && "Expected exactly one fix.");
    for (const auto &Edit : Diag.Fixes.front().Edits) {
      Edits.try_emplace(Edit.newText, Edit);
    }
  }
  for (auto &It : Edits)
    AddAllMissing.Edits.push_back(std::move(It.second));
  return AddAllMissing;
}
Fix fixAll(const Fix &RemoveAllUnused, const Fix &AddAllMissing) {
  Fix FixAll;
  FixAll.Message = "fix all includes";

  for (const auto &F : RemoveAllUnused.Edits)
    FixAll.Edits.push_back(F);
  for (const auto &F : AddAllMissing.Edits)
    FixAll.Edits.push_back(F);
  return FixAll;
}

std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles,
          bool AnalyzeAngledIncludes) {
  trace::Span Tracer("IncludeCleaner::getUnused");
  std::vector<const Inclusion *> Unused;
  for (const Inclusion &MFI : AST.getIncludeStructure().MainFileIncludes) {
    if (!MFI.HeaderID)
      continue;
    auto IncludeID = static_cast<IncludeStructure::HeaderID>(*MFI.HeaderID);
    if (ReferencedFiles.contains(IncludeID))
      continue;
    if (!mayConsiderUnused(MFI, AST, &AST.getPragmaIncludes(),
                           AnalyzeAngledIncludes)) {
      dlog("{0} was not used, but is not eligible to be diagnosed as unused",
           MFI.Written);
      continue;
    }
    Unused.push_back(&MFI);
  }
  return Unused;
}

} // namespace

std::vector<include_cleaner::SymbolReference>
collectMacroReferences(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();
  auto &PP = AST.getPreprocessor();
  std::vector<include_cleaner::SymbolReference> Macros;
  for (const auto &[_, Refs] : AST.getMacros().MacroRefs) {
    for (const auto &Ref : Refs) {
      auto Loc = SM.getComposedLoc(SM.getMainFileID(), Ref.StartOffset);
      const auto *Tok = AST.getTokens().spelledTokenContaining(Loc);
      if (!Tok)
        continue;
      auto Macro = locateMacroAt(*Tok, PP);
      if (!Macro)
        continue;
      auto DefLoc = Macro->NameLoc;
      if (!DefLoc.isValid())
        continue;
      Macros.push_back(
          {include_cleaner::Macro{/*Name=*/PP.getIdentifierInfo(Tok->text(SM)),
                                  DefLoc},
           Tok->location(),
           Ref.InConditionalDirective ? include_cleaner::RefType::Ambiguous
                                      : include_cleaner::RefType::Explicit});
    }
  }

  return Macros;
}

include_cleaner::Includes convertIncludes(const ParsedAST &AST) {
  auto &SM = AST.getSourceManager();

  include_cleaner::Includes ConvertedIncludes;
  // We satisfy Includes's contract that search dirs and included files have
  // matching path styles: both ultimately use FileManager::getCanonicalName().
  for (const auto &Dir : AST.getIncludeStructure().SearchPathsCanonical)
    ConvertedIncludes.addSearchDirectory(Dir);

  for (const Inclusion &Inc : AST.getIncludeStructure().MainFileIncludes) {
    include_cleaner::Include TransformedInc;
    llvm::StringRef WrittenRef = llvm::StringRef(Inc.Written);
    TransformedInc.Spelled = WrittenRef.trim("\"<>");
    TransformedInc.HashLocation =
        SM.getComposedLoc(SM.getMainFileID(), Inc.HashOffset);
    TransformedInc.Line = Inc.HashLine + 1;
    TransformedInc.Angled = WrittenRef.starts_with("<");
    // Inc.Resolved is canonicalized with clangd::getCanonicalPath(),
    // which is based on FileManager::getCanonicalName(ParentDir).
    auto FE = SM.getFileManager().getFileRef(Inc.Resolved);
    if (!FE) {
      elog("IncludeCleaner: Failed to get an entry for resolved path {0}: {1}",
           Inc.Resolved, FE.takeError());
      continue;
    }
    TransformedInc.Resolved = *FE;
    ConvertedIncludes.add(std::move(TransformedInc));
  }
  return ConvertedIncludes;
}

IncludeCleanerFindings
computeIncludeCleanerFindings(ParsedAST &AST, bool AnalyzeAngledIncludes) {
  // Interaction is only polished for C/CPP.
  if (AST.getLangOpts().ObjC)
    return {};
  const auto &SM = AST.getSourceManager();
  include_cleaner::Includes ConvertedIncludes = convertIncludes(AST);
  const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID());
  auto PreamblePatch = PreamblePatch::getPatchEntry(AST.tuPath(), SM);

  std::vector<include_cleaner::SymbolReference> Macros =
      collectMacroReferences(AST);
  std::vector<MissingIncludeDiagInfo> MissingIncludes;
  llvm::DenseSet<IncludeStructure::HeaderID> Used;
  trace::Span Tracer("include_cleaner::walkUsed");
  OptionalDirectoryEntryRef ResourceDir = AST.getPreprocessor()
                                              .getHeaderSearchInfo()
                                              .getModuleMap()
                                              .getBuiltinDir();
  include_cleaner::walkUsed(
      AST.getLocalTopLevelDecls(), /*MacroRefs=*/Macros,
      &AST.getPragmaIncludes(), AST.getPreprocessor(),
      [&](const include_cleaner::SymbolReference &Ref,
          llvm::ArrayRef<include_cleaner::Header> Providers) {
        bool Satisfied = false;
        for (const auto &H : Providers) {
          if (H.kind() == include_cleaner::Header::Physical &&
              (H.physical() == MainFile || H.physical() == PreamblePatch ||
               H.physical().getDir() == ResourceDir)) {
            Satisfied = true;
            continue;
          }
          for (auto *Inc : ConvertedIncludes.match(H)) {
            Satisfied = true;
            auto HeaderID =
                AST.getIncludeStructure().getID(&Inc->Resolved->getFileEntry());
            assert(HeaderID.has_value() &&
                   "ConvertedIncludes only contains resolved includes.");
            Used.insert(*HeaderID);
          }
        }

        if (Satisfied || Providers.empty() ||
            Ref.RT != include_cleaner::RefType::Explicit)
          return;

        // Check if we have any headers with the same spelling, in edge cases
        // like `#include_next "foo.h"`, the user can't ever include the
        // physical foo.h, but can have a spelling that refers to it.
        // We postpone this check because spelling a header for every usage is
        // expensive.
        std::string Spelling = include_cleaner::spellHeader(
            {Providers.front(), AST.getPreprocessor().getHeaderSearchInfo(),
             MainFile});
        for (auto *Inc :
             ConvertedIncludes.match(include_cleaner::Header{Spelling})) {
          Satisfied = true;
          auto HeaderID =
              AST.getIncludeStructure().getID(&Inc->Resolved->getFileEntry());
          assert(HeaderID.has_value() &&
                 "ConvertedIncludes only contains resolved includes.");
          Used.insert(*HeaderID);
        }
        if (Satisfied)
          return;

        // We actually always want to map usages to their spellings, but
        // spelling locations can point into preamble section. Using these
        // offsets could lead into crashes in presence of stale preambles. Hence
        // we use "getFileLoc" instead to make sure it always points into main
        // file.
        // FIXME: Use presumed locations to map such usages back to patched
        // locations safely.
        auto Loc = SM.getFileLoc(Ref.RefLocation);
        // File locations can be outside of the main file if macro is expanded
        // through an #include.
        while (SM.getFileID(Loc) != SM.getMainFileID())
          Loc = SM.getIncludeLoc(SM.getFileID(Loc));
        auto TouchingTokens =
            syntax::spelledTokensTouching(Loc, AST.getTokens());
        assert(!TouchingTokens.empty());
        // Loc points to the start offset of the ref token, here we use the last
        // element of the TouchingTokens, e.g. avoid getting the "::" for
        // "ns::^abc".
        MissingIncludeDiagInfo DiagInfo{
            Ref.Target, TouchingTokens.back().range(SM), Providers};
        MissingIncludes.push_back(std::move(DiagInfo));
      });
  // Put possibly equal diagnostics together for deduplication.
  // The duplicates might be from macro arguments that get expanded multiple
  // times.
  llvm::stable_sort(MissingIncludes, [](const MissingIncludeDiagInfo &LHS,
                                        const MissingIncludeDiagInfo &RHS) {
    // First sort by reference location.
    if (LHS.SymRefRange != RHS.SymRefRange) {
      // We can get away just by comparing the offsets as all the ranges are in
      // main file.
      return LHS.SymRefRange.beginOffset() < RHS.SymRefRange.beginOffset();
    }
    // For the same location, break ties using the symbol. Note that this won't
    // be stable across runs.
    using MapInfo = llvm::DenseMapInfo<include_cleaner::Symbol>;
    return MapInfo::getHashValue(LHS.Symbol) <
           MapInfo::getHashValue(RHS.Symbol);
  });
  MissingIncludes.erase(llvm::unique(MissingIncludes), MissingIncludes.end());
  std::vector<const Inclusion *> UnusedIncludes =
      getUnused(AST, Used, AnalyzeAngledIncludes);
  return {std::move(UnusedIncludes), std::move(MissingIncludes)};
}

bool isPreferredProvider(const Inclusion &Inc,
                         const include_cleaner::Includes &Includes,
                         llvm::ArrayRef<include_cleaner::Header> Providers) {
  for (const auto &H : Providers) {
    auto Matches = Includes.match(H);
    for (const include_cleaner::Include *Match : Matches)
      if (Match->Line == unsigned(Inc.HashLine + 1))
        return true; // this header is (equal) best
    if (!Matches.empty())
      return false; // another header is better
  }
  return false; // no header provides the symbol
}

std::vector<Diag>
issueIncludeCleanerDiagnostics(ParsedAST &AST, llvm::StringRef Code,
                               const IncludeCleanerFindings &Findings,
                               const ThreadsafeFS &TFS,
                               HeaderFilter IgnoreHeaders) {
  trace::Span Tracer("IncludeCleaner::issueIncludeCleanerDiagnostics");
  std::vector<Diag> UnusedIncludes = generateUnusedIncludeDiagnostics(
      AST.tuPath(), Findings.UnusedIncludes, Code, IgnoreHeaders);
  std::optional<Fix> RemoveAllUnused = removeAllUnusedIncludes(UnusedIncludes);

  std::vector<Diag> MissingIncludeDiags = generateMissingIncludeDiagnostics(
      AST, Findings.MissingIncludes, Code, IgnoreHeaders, TFS);
  std::optional<Fix> AddAllMissing = addAllMissingIncludes(MissingIncludeDiags);

  std::optional<Fix> FixAll;
  if (RemoveAllUnused && AddAllMissing)
    FixAll = fixAll(*RemoveAllUnused, *AddAllMissing);

  auto AddBatchFix = [](const std::optional<Fix> &F, clang::clangd::Diag *Out) {
    if (!F)
      return;
    Out->Fixes.push_back(*F);
  };
  for (auto &Diag : MissingIncludeDiags) {
    AddBatchFix(MissingIncludeDiags.size() > 1 ? AddAllMissing : std::nullopt,
                &Diag);
    AddBatchFix(FixAll, &Diag);
  }
  for (auto &Diag : UnusedIncludes) {
    AddBatchFix(UnusedIncludes.size() > 1 ? RemoveAllUnused : std::nullopt,
                &Diag);
    AddBatchFix(FixAll, &Diag);
  }

  auto Result = std::move(MissingIncludeDiags);
  llvm::move(UnusedIncludes, std::back_inserter(Result));
  return Result;
}

} // namespace clang::clangd
