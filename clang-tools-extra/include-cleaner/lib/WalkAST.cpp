//===--- WalkAST.cpp - Find declaration references in the AST -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
namespace include_cleaner {
namespace {
using DeclCallback = llvm::function_ref<void(SourceLocation, NamedDecl &)>;

class ASTWalker : public RecursiveASTVisitor<ASTWalker> {
  DeclCallback Callback;

  void report(SourceLocation Loc, NamedDecl *ND) {
    if (!ND || Loc.isInvalid())
      return;
    Callback(Loc, *cast<NamedDecl>(ND->getCanonicalDecl()));
  }

public:
  ASTWalker(DeclCallback Callback) : Callback(Callback) {}

  bool VisitTagTypeLoc(TagTypeLoc TTL) {
    report(TTL.getNameLoc(), TTL.getDecl());
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    report(DRE->getLocation(), DRE->getFoundDecl());
    return true;
  }
};

} // namespace

void walkAST(Decl &Root, DeclCallback Callback) {
  ASTWalker(Callback).TraverseDecl(&Root);
}

} // namespace include_cleaner
} // namespace clang
