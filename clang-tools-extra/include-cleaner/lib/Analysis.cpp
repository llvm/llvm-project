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
              const PragmaIncludes *PI, const SourceManager &SM,
              UsedSymbolCB CB) {
  // This is duplicated in writeHTMLReport, changes should be mirrored there.
  tooling::stdlib::Recognizer Recognizer;
  for (auto *Root : ASTRoots) {
    auto &SM = Root->getASTContext().getSourceManager();
    walkAST(*Root, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
      SymbolReference SymRef{Loc, ND, RT};
      if (auto SS = Recognizer(&ND)) {
        // FIXME: Also report forward decls from main-file, so that the caller
        // can decide to insert/ignore a header.
        return CB(SymRef, findHeaders(*SS, SM, PI));
      }
      // FIXME: Extract locations from redecls.
      return CB(SymRef, findHeaders(ND.getLocation(), SM, PI));
    });
  }
  for (const SymbolReference &MacroRef : MacroRefs) {
    assert(MacroRef.Target.kind() == Symbol::Macro);
    return CB(MacroRef,
              findHeaders(MacroRef.Target.macro().Definition, SM, PI));
  }
}

} // namespace clang::include_cleaner
