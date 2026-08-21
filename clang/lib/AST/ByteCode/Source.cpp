//===--- Source.cpp - Source expression tracking ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Source.h"
#include "clang/AST/Expr.h"

using namespace clang;
using namespace clang::interp;

SourceLocation SourceInfo::getLoc() const {
  if (const Expr *E = asExpr())
    return E->getExprLoc();
  if (const Stmt *S = asStmt())
    return S->getBeginLoc();
  if (const Decl *D = asDecl())
    return D->getBeginLoc();
  return SourceLocation();
}

SourceRange SourceInfo::getRange() const {
  if (const Expr *E = asExpr())
    return E->getSourceRange();
  if (const Stmt *S = asStmt())
    return S->getSourceRange();
  if (const Decl *D = asDecl())
    return D->getSourceRange();
  return SourceRange();
}

const Expr *SourceMapper::getExpr(CodePtr PC) const {
  return getSource(PC).asExpr();
}

SourceLocation SourceMapper::getLocation(CodePtr PC) const {
  return getSource(PC).getLoc();
}

SourceRange SourceMapper::getRange(CodePtr PC) const {
  return getSource(PC).getRange();
}
