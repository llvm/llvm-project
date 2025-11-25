//===-- SemaExpand.cpp - Semantic Analysis for Expansion Statements--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++26 expansion statements,
//  aka 'template for'.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"

using namespace clang;
using namespace sema;


CXXExpansionStmtDecl *
Sema::ActOnCXXExpansionStmtDecl(unsigned TemplateDepth,
                                SourceLocation TemplateKWLoc) {
  llvm_unreachable("TODO");
}

CXXExpansionStmtDecl *
Sema::BuildCXXExpansionStmtDecl(DeclContext *Ctx, SourceLocation TemplateKWLoc,
                                NonTypeTemplateParmDecl *NTTP) {
  llvm_unreachable("TODO");
}

ExprResult Sema::ActOnCXXExpansionInitList(MultiExprArg SubExprs,
                                           SourceLocation LBraceLoc,
                                           SourceLocation RBraceLoc) {
  return CXXExpansionInitListExpr::Create(Context, SubExprs, LBraceLoc,
                                          RBraceLoc);
}

StmtResult Sema::ActOnCXXExpansionStmtPattern(
    CXXExpansionStmtDecl *ESD, Stmt *Init, Stmt *ExpansionVarStmt,
    Expr *ExpansionInitializer, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  llvm_unreachable("TODO");
}

StmtResult Sema::FinishCXXExpansionStmt(Stmt *Exp, Stmt *Body) {
  llvm_unreachable("TODO");
}
