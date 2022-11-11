//===--- Analysis.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
#include "AnalysisInternal.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::include_cleaner {

void walkUsed(llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs,
              const PragmaIncludes &PI, const SourceManager &SM,
              UsedSymbolCB CB) {
  tooling::stdlib::Recognizer Recognizer;
  for (auto *Root : ASTRoots) {
    auto &SM = Root->getASTContext().getSourceManager();
    walkAST(*Root, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
      if (auto SS = Recognizer(&ND)) {
        // FIXME: Also report forward decls from main-file, so that the caller
        // can decide to insert/ignore a header.
        return CB({Loc, Symbol(*SS), RT}, findIncludeHeaders(*SS, SM, PI));
      }
      // FIXME: Extract locations from redecls.
      return CB({Loc, Symbol(ND), RT},
                findIncludeHeaders(ND.getLocation(), SM, PI));
    });
  }
  for (const SymbolReference &MacroRef : MacroRefs) {
    assert(MacroRef.Target.kind() == Symbol::Macro);
    // FIXME: Handle macro locations.
    return CB(MacroRef,
              findIncludeHeaders(MacroRef.Target.macro().Definition, SM, PI));
  }
}

llvm::SmallVector<Header> findIncludeHeaders(const SymbolLocation &SLoc,
                                             const SourceManager &SM,
                                             const PragmaIncludes &PI) {
  llvm::SmallVector<Header> Results;
  if (auto *Loc = std::get_if<SourceLocation>(&SLoc)) {
    // FIXME: Handle non self-contained files.
    FileID FID = SM.getFileID(*Loc);
    const auto *FE = SM.getFileEntryForID(FID);
    if (!FE)
      return {};

    // We treat the spelling header in the IWYU pragma as the final public
    // header.
    // FIXME: look for exporters if the public header is exported by another
    // header.
    llvm::StringRef VerbatimSpelling = PI.getPublic(FE);
    if (!VerbatimSpelling.empty())
      return {Header(VerbatimSpelling)};

    Results = {Header(FE)};
    // FIXME: compute transitive exporter headers.
    for (const auto *Export : PI.getExporters(FE, SM.getFileManager()))
      Results.push_back(Export);
    return Results;
  }
  if (auto *Sym = std::get_if<tooling::stdlib::Symbol>(&SLoc)) {
    for (const auto &H : Sym->headers())
      Results.push_back(H);
    return Results;
  }
  llvm_unreachable("unhandled SymbolLocation kind!");
}

} // namespace clang::include_cleaner
