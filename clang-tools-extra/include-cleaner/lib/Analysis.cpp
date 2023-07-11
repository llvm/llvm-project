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
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

namespace clang::include_cleaner {

void walkUsed(llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs,
              const PragmaIncludes *PI, const SourceManager &SM,
              UsedSymbolCB CB) {
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
    if (!SM.isWrittenInMainFile(SM.getSpellingLoc(MacroRef.RefLocation)))
      continue;
    CB(MacroRef, headersForSymbol(MacroRef.Target, SM, PI));
  }
}

AnalysisResults
analyze(llvm::ArrayRef<Decl *> ASTRoots,
        llvm::ArrayRef<SymbolReference> MacroRefs, const Includes &Inc,
        const PragmaIncludes *PI, const SourceManager &SM,
        const HeaderSearch &HS,
        llvm::function_ref<bool(llvm::StringRef)> HeaderFilter) {
  const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID());
  llvm::DenseSet<const Include *> Used;
  llvm::StringSet<> Missing;
  if (!HeaderFilter)
    HeaderFilter = [](llvm::StringRef) { return false; };
  walkUsed(ASTRoots, MacroRefs, PI, SM,
           [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
             bool Satisfied = false;
             for (const Header &H : Providers) {
               if (H.kind() == Header::Physical && H.physical() == MainFile)
                 Satisfied = true;
               for (const Include *I : Inc.match(H)) {
                 Used.insert(I);
                 Satisfied = true;
               }
             }
             if (!Satisfied && !Providers.empty() &&
                 Ref.RT == RefType::Explicit &&
                 !HeaderFilter(Providers.front().resolvedPath()))
               Missing.insert(spellHeader({Providers.front(), HS, MainFile}));
           });

  AnalysisResults Results;
  for (const Include &I : Inc.all()) {
    if (Used.contains(&I) || !I.Resolved ||
        HeaderFilter(I.Resolved->tryGetRealPathName()))
      continue;
    if (PI) {
      if (PI->shouldKeep(I.Line))
        continue;
      // Check if main file is the public interface for a private header. If so
      // we shouldn't diagnose it as unused.
      if (auto PHeader = PI->getPublic(I.Resolved); !PHeader.empty()) {
        PHeader = PHeader.trim("<>\"");
        // Since most private -> public mappings happen in a verbatim way, we
        // check textually here. This might go wrong in presence of symlinks or
        // header mappings. But that's not different than rest of the places.
        if (MainFile->tryGetRealPathName().endswith(PHeader))
          continue;
      }
    }
    Results.Unused.push_back(&I);
  }
  for (llvm::StringRef S : Missing.keys())
    Results.Missing.push_back(S.str());
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
  for (llvm::StringRef Spelled : Results.Missing)
    cantFail(R.add(tooling::Replacement(FileName, UINT_MAX, 0,
                                        ("#include " + Spelled).str())));
  // "cleanup" actually turns the UINT_MAX replacements into concrete edits.
  auto Positioned = cantFail(format::cleanupAroundReplacements(Code, R, Style));
  return cantFail(tooling::applyAllReplacements(Code, Positioned));
}

} // namespace clang::include_cleaner
