//===---- OpenACCClause.cpp - Classes for OpenACC Clauses  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclasses of the OpenACCClause class declared in
// OpenACCClause.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OpenACCClause.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

using namespace clang;

OpenACCDefaultClause *OpenACCDefaultClause::Create(const ASTContext &C,
                                                   OpenACCDefaultClauseKind K,
                                                   SourceLocation BeginLoc,
                                                   SourceLocation LParenLoc,
                                                   SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCDefaultClause), alignof(OpenACCDefaultClause));

  return new (Mem) OpenACCDefaultClause(K, BeginLoc, LParenLoc, EndLoc);
}

OpenACCIfClause *OpenACCIfClause::Create(const ASTContext &C,
                                         SourceLocation BeginLoc,
                                         SourceLocation LParenLoc,
                                         Expr *ConditionExpr,
                                         SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCIfClause), alignof(OpenACCIfClause));
  return new (Mem) OpenACCIfClause(BeginLoc, LParenLoc, ConditionExpr, EndLoc);
}

OpenACCIfClause::OpenACCIfClause(SourceLocation BeginLoc,
                                 SourceLocation LParenLoc, Expr *ConditionExpr,
                                 SourceLocation EndLoc)
    : OpenACCClauseWithCondition(OpenACCClauseKind::If, BeginLoc, LParenLoc,
                                 ConditionExpr, EndLoc) {
  assert(ConditionExpr && "if clause requires condition expr");
  assert((ConditionExpr->isInstantiationDependent() ||
          ConditionExpr->getType()->isScalarType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCSelfClause *OpenACCSelfClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             Expr *ConditionExpr,
                                             SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCIfClause), alignof(OpenACCIfClause));
  return new (Mem)
      OpenACCSelfClause(BeginLoc, LParenLoc, ConditionExpr, EndLoc);
}

OpenACCSelfClause::OpenACCSelfClause(SourceLocation BeginLoc,
                                     SourceLocation LParenLoc,
                                     Expr *ConditionExpr, SourceLocation EndLoc)
    : OpenACCClauseWithCondition(OpenACCClauseKind::Self, BeginLoc, LParenLoc,
                                 ConditionExpr, EndLoc) {
  assert((!ConditionExpr || ConditionExpr->isInstantiationDependent() ||
          ConditionExpr->getType()->isScalarType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCClause::child_range OpenACCClause::children() {
  switch (getClauseKind()) {
  default:
    assert(false && "Clause children function not implemented");
    break;
#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  case OpenACCClauseKind::CLAUSE_NAME:                                         \
    return cast<OpenACC##CLAUSE_NAME##Clause>(this)->children();

#include "clang/Basic/OpenACCClauses.def"
  }
  return child_range(child_iterator(), child_iterator());
}

//===----------------------------------------------------------------------===//
//  OpenACC clauses printing methods
//===----------------------------------------------------------------------===//
void OpenACCClausePrinter::VisitDefaultClause(const OpenACCDefaultClause &C) {
  OS << "default(" << C.getDefaultClauseKind() << ")";
}

void OpenACCClausePrinter::VisitIfClause(const OpenACCIfClause &C) {
  OS << "if(" << C.getConditionExpr() << ")";
}

void OpenACCClausePrinter::VisitSelfClause(const OpenACCSelfClause &C) {
  OS << "self";
  if (const Expr *CondExpr = C.getConditionExpr())
    OS << "(" << CondExpr << ")";
}
