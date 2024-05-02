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
#define CLAUSE_ALIAS(ALIAS_NAME, CLAUSE_NAME)                                  \
  case OpenACCClauseKind::ALIAS_NAME:                                          \
    return cast<OpenACC##CLAUSE_NAME##Clause>(this)->children();

#include "clang/Basic/OpenACCClauses.def"
  }
  return child_range(child_iterator(), child_iterator());
}

OpenACCNumWorkersClause::OpenACCNumWorkersClause(SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 Expr *IntExpr,
                                                 SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::NumWorkers, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCNumWorkersClause *
OpenACCNumWorkersClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                                SourceLocation LParenLoc, Expr *IntExpr,
                                SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCNumWorkersClause),
                         alignof(OpenACCNumWorkersClause));
  return new (Mem)
      OpenACCNumWorkersClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCVectorLengthClause::OpenACCVectorLengthClause(SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     Expr *IntExpr,
                                                     SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::VectorLength, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCVectorLengthClause *
OpenACCVectorLengthClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                                  SourceLocation LParenLoc, Expr *IntExpr,
                                  SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCVectorLengthClause),
                         alignof(OpenACCVectorLengthClause));
  return new (Mem)
      OpenACCVectorLengthClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCNumGangsClause *OpenACCNumGangsClause::Create(const ASTContext &C,
                                                     SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     ArrayRef<Expr *> IntExprs,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCNumGangsClause::totalSizeToAlloc<Expr *>(IntExprs.size()));
  return new (Mem) OpenACCNumGangsClause(BeginLoc, LParenLoc, IntExprs, EndLoc);
}

OpenACCPrivateClause *OpenACCPrivateClause::Create(const ASTContext &C,
                                                   SourceLocation BeginLoc,
                                                   SourceLocation LParenLoc,
                                                   ArrayRef<Expr *> VarList,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCPrivateClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCPrivateClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCFirstPrivateClause *OpenACCFirstPrivateClause::Create(
    const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
    ArrayRef<Expr *> VarList, SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCFirstPrivateClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem)
      OpenACCFirstPrivateClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCNoCreateClause *OpenACCNoCreateClause::Create(const ASTContext &C,
                                                     SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     ArrayRef<Expr *> VarList,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCNoCreateClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCNoCreateClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCPresentClause *OpenACCPresentClause::Create(const ASTContext &C,
                                                   SourceLocation BeginLoc,
                                                   SourceLocation LParenLoc,
                                                   ArrayRef<Expr *> VarList,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCPresentClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCPresentClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCCopyClause *
OpenACCCopyClause::Create(const ASTContext &C, OpenACCClauseKind Spelling,
                          SourceLocation BeginLoc, SourceLocation LParenLoc,
                          ArrayRef<Expr *> VarList, SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCCopyClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem)
      OpenACCCopyClause(Spelling, BeginLoc, LParenLoc, VarList, EndLoc);
}

//===----------------------------------------------------------------------===//
//  OpenACC clauses printing methods
//===----------------------------------------------------------------------===//

void OpenACCClausePrinter::printExpr(const Expr *E) {
  E->printPretty(OS, nullptr, Policy, 0);
}

void OpenACCClausePrinter::VisitDefaultClause(const OpenACCDefaultClause &C) {
  OS << "default(" << C.getDefaultClauseKind() << ")";
}

void OpenACCClausePrinter::VisitIfClause(const OpenACCIfClause &C) {
  OS << "if(";
  printExpr(C.getConditionExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitSelfClause(const OpenACCSelfClause &C) {
  OS << "self";
  if (const Expr *CondExpr = C.getConditionExpr()) {
    OS << "(";
    printExpr(CondExpr);
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitNumGangsClause(const OpenACCNumGangsClause &C) {
  OS << "num_gangs(";
  llvm::interleaveComma(C.getIntExprs(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitNumWorkersClause(
    const OpenACCNumWorkersClause &C) {
  OS << "num_workers(";
  printExpr(C.getIntExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitVectorLengthClause(
    const OpenACCVectorLengthClause &C) {
  OS << "vector_length(";
  printExpr(C.getIntExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitPrivateClause(const OpenACCPrivateClause &C) {
  OS << "private(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitFirstPrivateClause(
    const OpenACCFirstPrivateClause &C) {
  OS << "firstprivate(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitNoCreateClause(const OpenACCNoCreateClause &C) {
  OS << "no_create(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitPresentClause(const OpenACCPresentClause &C) {
  OS << "present(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitCopyClause(const OpenACCCopyClause &C) {
  OS << C.getClauseKind() << '(';
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}
