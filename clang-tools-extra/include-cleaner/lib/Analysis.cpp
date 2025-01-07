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
  for (auto *Root : ASTRoots) {
    walkAST(*Root, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
      auto FID = SM.getFileID(SM.getSpellingLoc(Loc));
      if (FID != SM.getMainFileID() && FID != SM.getPreambleFileID())
        return;
      // FIXME: Most of the work done here is repetitive. It might be useful to
      // have a cache/batching.
      SymbolReference SymRef{ND, Loc, RT};
      return CB(SymRef, headersForSymbol(ND, SM, PI));
    });
  }
  for (const SymbolReference &MacroRef : MacroRefs) {
    assert(MacroRef.Target.kind() == Symbol::Macro);
    if (!SM.isWrittenInMainFile(SM.getSpellingLoc(MacroRef.RefLocation)) ||
        shouldIgnoreMacroReference(PP, MacroRef.Target.macro()))
      continue;
    CB(MacroRef, headersForSymbol(MacroRef.Target, SM, PI));
  }
}

AnalysisResults
analyze(llvm::ArrayRef<Decl *> ASTRoots,
        llvm::ArrayRef<SymbolReference> MacroRefs, const Includes &Inc,
        const PragmaIncludes *PI, const Preprocessor &PP,
        llvm::function_ref<bool(llvm::StringRef)> HeaderFilter) {
  auto &SM = PP.getSourceManager();
  const auto MainFile = *SM.getFileEntryRefForID(SM.getMainFileID());
  llvm::DenseSet<const Include *> Used;
  llvm::StringMap<Header> Missing;
  if (!HeaderFilter)
    HeaderFilter = [](llvm::StringRef) { return false; };
  OptionalDirectoryEntryRef ResourceDir =
      PP.getHeaderSearchInfo().getModuleMap().getBuiltinDir();
  walkUsed(ASTRoots, MacroRefs, PI, PP,
           [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
             bool Satisfied = false;
             for (const Header &H : Providers) {
               if (H.kind() == Header::Physical &&
                   (H.physical() == MainFile ||
                    H.physical().getDir() == ResourceDir)) {
                 Satisfied = true;
               }
               for (const Include *I : Inc.match(H)) {
                 Used.insert(I);
                 Satisfied = true;
               }
             }
             // Bail out if we can't (or need not) insert an include.
             if (Satisfied || Providers.empty() || Ref.RT != RefType::Explicit)
               return;
             if (HeaderFilter(Providers.front().resolvedPath()))
               return;
             // Check if we have any headers with the same spelling, in edge
             // cases like `#include_next "foo.h"`, the user can't ever
             // include the physical foo.h, but can have a spelling that
             // refers to it.
             auto Spelling = spellHeader(
                 {Providers.front(), PP.getHeaderSearchInfo(), MainFile});
             for (const Include *I : Inc.match(Header{Spelling})) {
               Used.insert(I);
               Satisfied = true;
             }
             if (!Satisfied)
               Missing.try_emplace(std::move(Spelling), Providers.front());
           });

  AnalysisResults Results;
  for (const Include &I : Inc.all()) {
    if (Used.contains(&I) || !I.Resolved ||
        HeaderFilter(I.Resolved->getName()) ||
        I.Resolved->getDir() == ResourceDir)
      continue;
    if (PI) {
      if (PI->shouldKeep(*I.Resolved))
        continue;
      // Check if main file is the public interface for a private header. If so
      // we shouldn't diagnose it as unused.
      if (auto PHeader = PI->getPublic(*I.Resolved); !PHeader.empty()) {
        PHeader = PHeader.trim("<>\"");
        // Since most private -> public mappings happen in a verbatim way, we
        // check textually here. This might go wrong in presence of symlinks or
        // header mappings. But that's not different than rest of the places.
        if (MainFile.getName().ends_with(PHeader))
          continue;
      }
    }
    Results.Unused.push_back(&I);
  }
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
