//===--- Analysis.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
#include "AnalysisInternal.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <climits>
#include <string>
#include <utility>

namespace clang::include_cleaner {

namespace {
bool shouldIgnoreMacroReference(const Preprocessor &PP, const Macro &M) {
  auto *MI = PP.getMacroInfo(M.Name);
  // Macros that expand to themselves are confusing from user's point of view.
  // They usually aspect the usage to be attributed to the underlying decl and
  // not the macro definition. So ignore such macros (e.g. std{in,out,err} are
  // implementation defined macros, that just resolve to themselves in
  // practice).
  return MI && MI->getNumTokens() == 1 && MI->isObjectLike() &&
         MI->getReplacementToken(0).getIdentifierInfo() == M.Name;
}
} // namespace

void walkUsed(llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs,
              const PragmaIncludes *PI, const Preprocessor &PP,
              UsedSymbolCB CB) {
  const auto &SM = PP.getSourceManager();
  // This is duplicated in writeHTMLReport, changes should be mirrored there.
  tooling::stdlib::Recognizer Recognizer;
  walkAST(ASTRoots, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
    auto SpellLoc = SM.getSpellingLoc(Loc);
    // Tokens resulting from macro concatenation ends up in scratch space and
    // clang currently doesn't have a good/simple APIs for tracking where
    // pieces of a concatenated token originated from.
    // So we use the macro expansion location instead, and downgrade reference
    // type to ambiguous to prevent false negatives.
    if (SM.isWrittenInScratchSpace(SpellLoc)) {
      Loc = SM.getExpansionLoc(Loc);
      if (RT == RefType::Explicit)
        RT = RefType::Ambiguous;
      SpellLoc = SM.getSpellingLoc(Loc);
    }
    auto FID = SM.getFileID(SpellLoc);
    if (FID != SM.getMainFileID() && FID != SM.getPreambleFileID())
      return;
    // FIXME: Most of the work done here is repetitive. It might be useful to
    // have a cache/batching.
    SymbolReference SymRef{ND, Loc, RT};
    return CB(SymRef, headersForSymbol(ND, PP, PI));
  });
  for (const SymbolReference &MacroRef : MacroRefs) {
    assert(MacroRef.Target.kind() == Symbol::Macro);
    if (!SM.isWrittenInMainFile(SM.getSpellingLoc(MacroRef.RefLocation)) ||
        shouldIgnoreMacroReference(PP, MacroRef.Target.macro()))
      continue;
    CB(MacroRef, headersForSymbol(MacroRef.Target, PP, PI));
  }
}

void analyze(
    llvm::ArrayRef<Decl *> ASTRoots, llvm::ArrayRef<SymbolReference> MacroRefs,
    const Includes &Inc, const PragmaIncludes *PI, const Preprocessor &PP,
    llvm::function_ref<bool(llvm::StringRef)> HeaderFilter,
    llvm::function_ref<void(const Include *)> UnusedHeaderCallback,
    llvm::function_ref<void(const Header *, const Include *,
                            llvm::StringRef Spelling, const SymbolReference &,
                            llvm::ArrayRef<Header> Providers)>
        Callback) {
  auto &SM = PP.getSourceManager();
  const auto MainFile = *SM.getFileEntryRefForID(SM.getMainFileID());
  llvm::DenseSet<const Include *> Used;
  constexpr auto DefaultHeaderFilter = [](llvm::StringRef) { return false; };
  if (!HeaderFilter)
    HeaderFilter = DefaultHeaderFilter;
  OptionalDirectoryEntryRef ResourceDir =
      PP.getHeaderSearchInfo().getModuleMap().getBuiltinDir();
  walkUsed(ASTRoots, MacroRefs, PI, PP,
           [=, &Used, &PP, &Inc](const SymbolReference &Ref,
                                 llvm::ArrayRef<Header> Providers) {
             std::pair<const Header *, const Include *> Satisfied;
             for (const Header &H : Providers) {
               if (H.kind() == Header::Physical &&
                   (H.physical() == MainFile ||
                    H.physical().getDir() == ResourceDir)) {
                 Satisfied = {&H, nullptr};
               }
               for (const Include *I : Inc.match(H)) {
                 Used.insert(I);
                 Satisfied = {&H, I};
               }
             }
             // Bail out if we can't (or need not) insert an include.
             if (Satisfied.first || Providers.empty() ||
                 Ref.RT != RefType::Explicit)
               return;
             if (HeaderFilter(Providers.front().resolvedPath()))
               return;
             // Check if we have any headers with the same spelling, in edge
             // cases like `#include_next "foo.h"`, the user can't ever
             // include the physical foo.h, but can have a spelling that
             // refers to it.
             auto Spelling = spellHeader(
                 {Providers.front(), PP.getHeaderSearchInfo(), MainFile});
             Header similar_header(Spelling);
             for (const Include *I : Inc.match(similar_header)) {
               Used.insert(I);
               Satisfied = {&similar_header, I};
             }
             if (Callback) {
               Callback(Satisfied.first, Satisfied.second, Spelling, Ref,
                        Providers);
             }
           });

  if (UnusedHeaderCallback) {
    for (const Include &I : Inc.all()) {
      if (Used.contains(&I) || !I.Resolved ||
          HeaderFilter(I.Resolved->getName()) ||
          I.Resolved->getDir() == ResourceDir)
        continue;
      if (PI) {
        if (PI->shouldKeep(*I.Resolved))
          continue;
        // Check if main file is the public interface for a private header. If
        // so we shouldn't diagnose it as unused.
        if (auto PHeader = PI->getPublic(*I.Resolved); !PHeader.empty()) {
          PHeader = PHeader.trim("<>\"");
          // Since most private -> public mappings happen in a verbatim way, we
          // check textually here. This might go wrong in presence of symlinks
          // or header mappings. But that's not different than rest of the
          // places.
          if (MainFile.getName().ends_with(PHeader))
            continue;
        }
      }
      UnusedHeaderCallback(&I);
    }
  }
}

AnalysisResults
analyze(llvm::ArrayRef<Decl *> ASTRoots,
        llvm::ArrayRef<SymbolReference> MacroRefs, const Includes &Inc,
        const PragmaIncludes *PI, const Preprocessor &PP,
        llvm::function_ref<bool(llvm::StringRef)> HeaderFilter) {
  llvm::StringMap<Header> Missing;
  AnalysisResults Results;
  analyze(
      ASTRoots, MacroRefs, Inc, PI, PP, HeaderFilter,
      [&Results](const Include *I) { Results.Unused.push_back(I); },
      [&Results](const Header *H, const Include *, llvm::StringRef Spelling,
                 const SymbolReference &, llvm::ArrayRef<Header> Providers) {
        if (!H) {
          Results.Missing.emplace_back(Spelling, Providers.front());
        }
      });
  for (auto &E : Missing)
    Results.Missing.emplace_back(E.first().str(), E.second);
  llvm::sort(Results.Missing);
  return Results;
}

std::string fixIncludes(const AnalysisResults &Results,
                        llvm::StringRef FileName, llvm::StringRef Code,
                        const format::FormatStyle &Style) {
  assert(Style.isCpp() && "Only C++ style supports include insertions!");
  tooling::Replacements R;
  // Encode insertions/deletions in the magic way clang-format understands.
  for (const Include *I : Results.Unused)
    cantFail(R.add(tooling::Replacement(FileName, UINT_MAX, 1, I->quote())));
  for (auto &[Spelled, _] : Results.Missing)
    cantFail(R.add(
        tooling::Replacement(FileName, UINT_MAX, 0, "#include " + Spelled)));
  // "cleanup" actually turns the UINT_MAX replacements into concrete edits.
  auto Positioned = cantFail(format::cleanupAroundReplacements(Code, R, Style));
  return cantFail(tooling::applyAllReplacements(Code, Positioned));
}

} // namespace clang::include_cleaner
