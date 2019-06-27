//===--- StmtUtils.cpp - Statement helper functions -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StmtUtils.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Lex/Lexer.h"

using namespace clang;

SourceLocation
clang::tooling::getLexicalEndLocForDecl(const Decl *D, const SourceManager &SM,
                                        const LangOptions &LangOpts) {
  if (!isa<ObjCImplDecl>(D))
    return D->getSourceRange().getEnd();
  auto AtEnd = D->getSourceRange().getEnd();
  auto AdjustedEnd =
      Lexer::findNextTokenLocationAfterTokenAt(AtEnd, SM, LangOpts);
  return AdjustedEnd.isValid() ? AdjustedEnd : AtEnd;
}

bool clang::tooling::isSemicolonRequiredAfter(const Stmt *S) {
  if (isa<CompoundStmt>(S))
    return false;
  if (const auto *If = dyn_cast<IfStmt>(S))
    return isSemicolonRequiredAfter(If->getElse() ? If->getElse()
                                                  : If->getThen());
  if (const auto *While = dyn_cast<WhileStmt>(S))
    return isSemicolonRequiredAfter(While->getBody());
  if (const auto *For = dyn_cast<ForStmt>(S))
    return isSemicolonRequiredAfter(For->getBody());
  if (const auto *CXXFor = dyn_cast<CXXForRangeStmt>(S))
    return isSemicolonRequiredAfter(CXXFor->getBody());
  if (const auto *ObjCFor = dyn_cast<ObjCForCollectionStmt>(S))
    return isSemicolonRequiredAfter(ObjCFor->getBody());
  switch (S->getStmtClass()) {
  case Stmt::SwitchStmtClass:
  case Stmt::CXXTryStmtClass:
  case Stmt::ObjCAtSynchronizedStmtClass:
  case Stmt::ObjCAutoreleasePoolStmtClass:
  case Stmt::ObjCAtTryStmtClass:
    return false;
  default:
    return true;
  }
}

static bool isAssignmentOperator(const Stmt *S) {
  if (const auto *PseudoExpr = dyn_cast<PseudoObjectExpr>(S))
    return isAssignmentOperator(PseudoExpr->getSyntacticForm());
  if (const auto *BO = dyn_cast<BinaryOperator>(S))
    return BO->isAssignmentOp();
  return false;
}

bool clang::tooling::isLexicalExpression(const Stmt *S, const Stmt *Parent) {
  if (!isa<Expr>(S))
    return false;
  // Assignment operators should be treated as statements unless they are a part
  // of an expression.
  if (isAssignmentOperator(S) && (!Parent || !isa<Expr>(Parent)))
    return false;
  return !cast<Expr>(S)->getType()->isVoidType();
}
