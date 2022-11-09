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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::include_cleaner {
namespace {
llvm::SmallVector<Header>
toHeader(llvm::ArrayRef<tooling::stdlib::Header> Headers) {
  llvm::SmallVector<Header> Result;
  llvm::for_each(Headers, [&](tooling::stdlib::Header H) {
    Result.emplace_back(Header(H));
  });
  return Result;
}
} // namespace

void walkUsed(const SourceManager &SM, llvm::ArrayRef<Decl *> ASTRoots,
              llvm::ArrayRef<SymbolReference> MacroRefs, UsedSymbolCB CB) {
  tooling::stdlib::Recognizer Recognizer;
  for (auto *Root : ASTRoots) {
    auto &SM = Root->getASTContext().getSourceManager();
    walkAST(*Root, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
      if (auto SS = Recognizer(&ND)) {
        // FIXME: Also report forward decls from main-file, so that the caller
        // can decide to insert/ignore a header.
        return CB({Loc, Symbol(*SS), RT}, toHeader(SS->headers()));
      }
      // FIXME: Extract locations from redecls.
      // FIXME: Handle IWYU pragmas, non self-contained files.
      // FIXME: Handle macro locations.
      if (auto *FE = SM.getFileEntryForID(SM.getFileID(ND.getLocation())))
        return CB({Loc, Symbol(ND), RT}, {Header(FE)});
    });
  }
  for (const SymbolReference &MacroRef : MacroRefs) {
    assert(MacroRef.Target.kind() == Symbol::Macro);
    // FIXME: Handle IWYU pragmas, non self-contained files.
    // FIXME: Handle macro locations.
    if (auto *FE = SM.getFileEntryForID(
            SM.getFileID(MacroRef.Target.macro().Definition)))
      CB(MacroRef, {Header(FE)});
  }
}

} // namespace clang::include_cleaner
