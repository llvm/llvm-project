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
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>

namespace clang {
namespace dataflow {

class TransferVisitor : public ConstStmtVisitor<TransferVisitor> {
public:
  TransferVisitor(Environment &Env) : Env(Env) {}

  void VisitBinaryOperator(const BinaryOperator *S) {
    if (S->getOpcode() == BO_Assign) {
      // The CFG does not contain `ParenExpr` as top-level statements in basic
      // blocks, however sub-expressions can still be of that type.
      assert(S->getLHS() != nullptr);
      const Expr *LHS = S->getLHS()->IgnoreParens();

      assert(LHS != nullptr);
      auto *LHSLoc = Env.getStorageLocation(*LHS, SkipPast::Reference);
      if (LHSLoc == nullptr)
        return;

      // The CFG does not contain `ParenExpr` as top-level statements in basic
      // blocks, however sub-expressions can still be of that type.
      assert(S->getRHS() != nullptr);
      const Expr *RHS = S->getRHS()->IgnoreParens();

      assert(RHS != nullptr);
      Value *RHSVal = Env.getValue(*RHS, SkipPast::Reference);
      if (RHSVal == nullptr)
        return;

      // Assign a value to the storage location of the left-hand side.
      Env.setValue(*LHSLoc, *RHSVal);

      // Assign a storage location for the whole expression.
      Env.setStorageLocation(*S, *LHSLoc);
    }
    // FIXME: Add support for BO_EQ, BO_NE.
  }

  void VisitDeclRefExpr(const DeclRefExpr *S) {
    assert(S->getDecl() != nullptr);
    auto *DeclLoc = Env.getStorageLocation(*S->getDecl(), SkipPast::None);
    if (DeclLoc == nullptr)
      return;

    if (S->getDecl()->getType()->isReferenceType()) {
      Env.setStorageLocation(*S, *DeclLoc);
    } else {
      auto &Loc = Env.createStorageLocation(*S);
      auto &Val = Env.takeOwnership(std::make_unique<ReferenceValue>(*DeclLoc));
      Env.setStorageLocation(*S, Loc);
      Env.setValue(Loc, Val);
    }
  }

  void VisitDeclStmt(const DeclStmt *S) {
    // FIXME: Add support for group decls, e.g: `int a, b;`
    if (S->isSingleDecl()) {
      if (const auto *D = dyn_cast<VarDecl>(S->getSingleDecl())) {
        visitVarDecl(*D);
      }
    }
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *S) {
    if (S->getCastKind() == CK_LValueToRValue) {
      // The CFG does not contain `ParenExpr` as top-level statements in basic
      // blocks, however sub-expressions can still be of that type.
      assert(S->getSubExpr() != nullptr);
      const Expr *SubExpr = S->getSubExpr()->IgnoreParens();

      assert(SubExpr != nullptr);
      auto *SubExprVal = Env.getValue(*SubExpr, SkipPast::Reference);
      if (SubExprVal == nullptr)
        return;

      auto &ExprLoc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, ExprLoc);
      Env.setValue(ExprLoc, *SubExprVal);
    }
    // FIXME: Add support for CK_NoOp, CK_UserDefinedConversion,
    // CK_ConstructorConversion, CK_UncheckedDerivedToBase.
  }

  void VisitUnaryOperator(const UnaryOperator *S) {
    if (S->getOpcode() == UO_Deref) {
      assert(S->getSubExpr() != nullptr);
      const auto *SubExprVal = cast_or_null<PointerValue>(
          Env.getValue(*S->getSubExpr(), SkipPast::Reference));
      if (SubExprVal == nullptr)
        return;

      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      Env.setValue(Loc, Env.takeOwnership(std::make_unique<ReferenceValue>(
                            SubExprVal->getPointeeLoc())));
    }
    // FIXME: Add support for UO_AddrOf, UO_LNot.
  }

  void VisitCXXThisExpr(const CXXThisExpr *S) {
    auto *ThisPointeeLoc = Env.getThisPointeeStorageLocation();
    assert(ThisPointeeLoc != nullptr);

    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    Env.setValue(Loc, Env.takeOwnership(
                          std::make_unique<PointerValue>(*ThisPointeeLoc)));
  }

  void VisitMemberExpr(const MemberExpr *S) {
    ValueDecl *Member = S->getMemberDecl();
    assert(Member != nullptr);

    // FIXME: Consider assigning pointer values to function member expressions.
    if (Member->isFunctionOrFunctionTemplate())
      return;

    // The receiver can be either a value or a pointer to a value. Skip past the
    // indirection to handle both cases.
    auto *BaseLoc = cast_or_null<AggregateStorageLocation>(
        Env.getStorageLocation(*S->getBase(), SkipPast::ReferenceThenPointer));
    if (BaseLoc == nullptr)
      return;

    // FIXME: Add support for union types.
    if (BaseLoc->getType()->isUnionType())
      return;

    auto &MemberLoc = BaseLoc->getChild(*Member);
    if (MemberLoc.getType()->isReferenceType()) {
      Env.setStorageLocation(*S, MemberLoc);
    } else {
      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      Env.setValue(
          Loc, Env.takeOwnership(std::make_unique<ReferenceValue>(MemberLoc)));
    }
  }

  // FIXME: Add support for:
  // - CallExpr
  // - CXXBindTemporaryExpr
  // - CXXBoolLiteralExpr
  // - CXXConstructExpr
  // - CXXFunctionalCastExpr
  // - CXXOperatorCallExpr
  // - CXXStaticCastExpr
  // - MaterializeTemporaryExpr

private:
  void visitVarDecl(const VarDecl &D) {
    auto &Loc = Env.createStorageLocation(D);
    Env.setStorageLocation(D, Loc);

    const Expr *InitExpr = D.getInit();
    if (InitExpr == nullptr) {
      // No initializer expression - associate `Loc` with a new value.
      Env.initValueInStorageLocation(Loc, D.getType());
      return;
    }

    if (D.getType()->isReferenceType()) {
      // Initializing a reference variable - do not create a reference to
      // reference.
      if (auto *InitExprLoc =
              Env.getStorageLocation(*InitExpr, SkipPast::Reference)) {
        auto &Val =
            Env.takeOwnership(std::make_unique<ReferenceValue>(*InitExprLoc));
        Env.setValue(Loc, Val);
      } else {
        // FIXME: The initializer expression must always be assigned a value.
        // Replace this with an assert when we have sufficient coverage of
        // language features.
        Env.initValueInStorageLocation(Loc, D.getType());
      }
      return;
    }

    if (auto *InitExprVal = Env.getValue(*InitExpr, SkipPast::None)) {
      Env.setValue(Loc, *InitExprVal);
    } else {
      // FIXME: The initializer expression must always be assigned a value.
      // Replace this with an assert when we have sufficient coverage of
      // language features.
      Env.initValueInStorageLocation(Loc, D.getType());
    }
  }

  Environment &Env;
};

void transfer(const Stmt &S, Environment &Env) {
  assert(!isa<ParenExpr>(&S));
  TransferVisitor(Env).Visit(&S);
}

} // namespace dataflow
} // namespace clang
