//===--- ParentMap.cpp - Mappings from Stmts to their Parents ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParentMap class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ParentMap.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/DenseMap.h"

using namespace clang;

typedef llvm::DenseMap<Stmt*, Stmt*> MapTy;

enum OpaqueValueMode {
  OV_Transparent,
  OV_Opaque
};

static void BuildParentMap(MapTy& M, Stmt* S,
                           OpaqueValueMode OVMode = OV_Transparent) {

  switch (S->getStmtClass()) {
  case Stmt::PseudoObjectExprClass: {
    assert(OVMode == OV_Transparent && "Should not appear alongside OVEs");
    PseudoObjectExpr *POE = cast<PseudoObjectExpr>(S);

    M[POE->getSyntacticForm()] = S;
    BuildParentMap(M, POE->getSyntacticForm(), OV_Transparent);

    for (PseudoObjectExpr::semantics_iterator I = POE->semantics_begin(),
                                              E = POE->semantics_end();
         I != E; ++I) {
      M[*I] = S;
      BuildParentMap(M, *I, OV_Opaque);
    }
    break;
  }
  case Stmt::BinaryConditionalOperatorClass: {
    assert(OVMode == OV_Transparent && "Should not appear alongside OVEs");
    BinaryConditionalOperator *BCO = cast<BinaryConditionalOperator>(S);

    M[BCO->getCommon()] = S;
    BuildParentMap(M, BCO->getCommon(), OV_Transparent);

    M[BCO->getCond()] = S;
    BuildParentMap(M, BCO->getCond(), OV_Opaque);

    M[BCO->getTrueExpr()] = S;
    BuildParentMap(M, BCO->getTrueExpr(), OV_Opaque);

    M[BCO->getFalseExpr()] = S;
    BuildParentMap(M, BCO->getFalseExpr(), OV_Transparent);

    break;
  }
  case Stmt::OpaqueValueExprClass:
    if (OVMode == OV_Transparent) {
      OpaqueValueExpr *OVE = cast<OpaqueValueExpr>(S);
      M[OVE->getSourceExpr()] = S;
      BuildParentMap(M, OVE->getSourceExpr(), OV_Transparent);
    }
    break;
  default:
    for (Stmt::child_range I = S->children(); I; ++I) {
      if (*I) {
        M[*I] = S;
        BuildParentMap(M, *I, OVMode);
      }
    }
    break;
  }
}

ParentMap::ParentMap(Stmt* S) : Impl(0) {
  if (S) {
    MapTy *M = new MapTy();
    BuildParentMap(*M, S);
    Impl = M;
  }
}

ParentMap::~ParentMap() {
  delete (MapTy*) Impl;
}

void ParentMap::addStmt(Stmt* S) {
  if (S) {
    BuildParentMap(*(MapTy*) Impl, S);
  }
}

Stmt* ParentMap::getParent(Stmt* S) const {
  MapTy* M = (MapTy*) Impl;
  MapTy::iterator I = M->find(S);
  return I == M->end() ? 0 : I->second;
}

Stmt *ParentMap::getParentIgnoreParens(Stmt *S) const {
  do { S = getParent(S); } while (S && isa<ParenExpr>(S));
  return S;
}

Stmt *ParentMap::getParentIgnoreParenCasts(Stmt *S) const {
  do {
    S = getParent(S);
  }
  while (S && (isa<ParenExpr>(S) || isa<CastExpr>(S)));

  return S;  
}

Stmt *ParentMap::getParentIgnoreParenImpCasts(Stmt *S) const {
  do {
    S = getParent(S);
  } while (S && isa<Expr>(S) && cast<Expr>(S)->IgnoreParenImpCasts() != S);

  return S;
}

Stmt *ParentMap::getOuterParenParent(Stmt *S) const {
  Stmt *Paren = 0;
  while (isa<ParenExpr>(S)) {
    Paren = S;
    S = getParent(S);
  };
  return Paren;
}

bool ParentMap::isConsumedExpr(Expr* E) const {
  Stmt *P = getParent(E);
  Stmt *DirectChild = E;

  // Ignore parents that are parentheses or casts.
  while (P && (isa<ParenExpr>(P) || isa<CastExpr>(P))) {
    DirectChild = P;
    P = getParent(P);
  }

  if (!P)
    return false;

  switch (P->getStmtClass()) {
    default:
      return isa<Expr>(P);
    case Stmt::DeclStmtClass:
      return true;
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *BE = cast<BinaryOperator>(P);
      // If it is a comma, only the right side is consumed.
      // If it isn't a comma, both sides are consumed.
      return BE->getOpcode()!=BO_Comma ||DirectChild==BE->getRHS();
    }
    case Stmt::ForStmtClass:
      return DirectChild == cast<ForStmt>(P)->getCond();
    case Stmt::WhileStmtClass:
      return DirectChild == cast<WhileStmt>(P)->getCond();
    case Stmt::DoStmtClass:
      return DirectChild == cast<DoStmt>(P)->getCond();
    case Stmt::IfStmtClass:
      return DirectChild == cast<IfStmt>(P)->getCond();
    case Stmt::IndirectGotoStmtClass:
      return DirectChild == cast<IndirectGotoStmt>(P)->getTarget();
    case Stmt::SwitchStmtClass:
      return DirectChild == cast<SwitchStmt>(P)->getCond();
    case Stmt::ReturnStmtClass:
      return true;
  }
}

