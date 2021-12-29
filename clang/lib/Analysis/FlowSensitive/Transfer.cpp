//===-- Transfer.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines transfer functions that evaluate program statements and
//  update an environment accordingly.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Transfer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "llvm/Support/Casting.h"
#include <cassert>

namespace clang {
namespace dataflow {

class TransferVisitor : public ConstStmtVisitor<TransferVisitor> {
public:
  TransferVisitor(Environment &Env) : Env(Env) {}

  void VisitDeclStmt(const DeclStmt *S) {
    // FIXME: Add support for group decls, e.g: `int a, b;`
    if (S->isSingleDecl()) {
      if (const auto *D = dyn_cast<VarDecl>(S->getSingleDecl())) {
        visitVarDecl(*D);
      }
    }
  }

  // FIXME: Add support for:
  // - BinaryOperator
  // - CallExpr
  // - CXXBindTemporaryExpr
  // - CXXBoolLiteralExpr
  // - CXXConstructExpr
  // - CXXFunctionalCastExpr
  // - CXXOperatorCallExpr
  // - CXXStaticCastExpr
  // - CXXThisExpr
  // - DeclRefExpr
  // - ImplicitCastExpr
  // - MaterializeTemporaryExpr
  // - MemberExpr
  // - UnaryOperator

private:
  void visitVarDecl(const VarDecl &D) {
    auto &Loc = Env.createStorageLocation(D);
    Env.setStorageLocation(D, Loc);
    Env.initValueInStorageLocation(Loc, D.getType());
  }

  Environment &Env;
};

void transfer(const Stmt &S, Environment &Env) {
  assert(!isa<ParenExpr>(&S));
  TransferVisitor(Env).Visit(&S);
}

} // namespace dataflow
} // namespace clang
