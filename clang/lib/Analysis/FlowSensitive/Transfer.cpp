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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>
#include <tuple>

namespace clang {
namespace dataflow {

static const Expr *skipExprWithCleanups(const Expr *E) {
  if (auto *C = dyn_cast_or_null<ExprWithCleanups>(E))
    return C->getSubExpr();
  return E;
}

class TransferVisitor : public ConstStmtVisitor<TransferVisitor> {
public:
  TransferVisitor(const StmtToEnvMap &StmtToEnv, Environment &Env)
      : StmtToEnv(StmtToEnv), Env(Env) {}

  void VisitBinaryOperator(const BinaryOperator *S) {
    switch (S->getOpcode()) {
    case BO_Assign: {
      // The CFG does not contain `ParenExpr` as top-level statements in basic
      // blocks, however sub-expressions can still be of that type.
      assert(S->getLHS() != nullptr);
      const Expr *LHS = S->getLHS()->IgnoreParens();

      assert(LHS != nullptr);
      auto *LHSLoc = Env.getStorageLocation(*LHS, SkipPast::Reference);
      if (LHSLoc == nullptr)
        break;

      // The CFG does not contain `ParenExpr` as top-level statements in basic
      // blocks, however sub-expressions can still be of that type.
      assert(S->getRHS() != nullptr);
      const Expr *RHS = S->getRHS()->IgnoreParens();

      assert(RHS != nullptr);
      Value *RHSVal = Env.getValue(*RHS, SkipPast::Reference);
      if (RHSVal == nullptr)
        break;

      // Assign a value to the storage location of the left-hand side.
      Env.setValue(*LHSLoc, *RHSVal);

      // Assign a storage location for the whole expression.
      Env.setStorageLocation(*S, *LHSLoc);
      break;
    }
    case BO_LAnd:
    case BO_LOr: {
      const Expr *LHS = S->getLHS();
      assert(LHS != nullptr);

      const Expr *RHS = S->getRHS();
      assert(RHS != nullptr);

      BoolValue *LHSVal =
          dyn_cast_or_null<BoolValue>(Env.getValue(*LHS, SkipPast::Reference));

      // `RHS` and `S` might be part of different basic blocks. We need to
      // access their values from the corresponding environments.
      BoolValue *RHSVal = nullptr;
      const Environment *RHSEnv = StmtToEnv.getEnvironment(*RHS);
      if (RHSEnv != nullptr)
        RHSVal = dyn_cast_or_null<BoolValue>(
            RHSEnv->getValue(*RHS, SkipPast::Reference));

      // Create fresh values for unknown boolean expressions.
      // FIXME: Consider providing a `GetOrCreateFresh` util in case this style
      // is expected to be common or make sure that all expressions are assigned
      // values and drop this.
      if (LHSVal == nullptr)
        LHSVal = &Env.takeOwnership(std::make_unique<AtomicBoolValue>());
      if (RHSVal == nullptr)
        RHSVal = &Env.takeOwnership(std::make_unique<AtomicBoolValue>());

      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      if (S->getOpcode() == BO_LAnd)
        Env.setValue(Loc, Env.takeOwnership(std::make_unique<ConjunctionValue>(
                              *LHSVal, *RHSVal)));
      else
        Env.setValue(Loc, Env.takeOwnership(std::make_unique<DisjunctionValue>(
                              *LHSVal, *RHSVal)));
      break;
    }
    default:
      // FIXME: Add support for BO_EQ, BO_NE.
      break;
    }
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
    // Group decls are converted into single decls in the CFG so the cast below
    // is safe.
    const auto &D = *cast<VarDecl>(S->getSingleDecl());
    auto &Loc = Env.createStorageLocation(D);
    Env.setStorageLocation(D, Loc);

    const Expr *InitExpr = D.getInit();
    if (InitExpr == nullptr) {
      // No initializer expression - associate `Loc` with a new value.
      if (Value *Val = Env.createValue(D.getType()))
        Env.setValue(Loc, *Val);
      return;
    }

    // The CFG does not contain `ParenExpr` as top-level statements in basic
    // blocks, however sub-expressions can still be of that type.
    InitExpr = skipExprWithCleanups(D.getInit()->IgnoreParens());
    assert(InitExpr != nullptr);

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
        if (Value *Val = Env.createValue(D.getType()))
          Env.setValue(Loc, *Val);
      }
      return;
    }

    if (auto *InitExprVal = Env.getValue(*InitExpr, SkipPast::None)) {
      Env.setValue(Loc, *InitExprVal);
    } else if (!D.getType()->isStructureOrClassType()) {
      // FIXME: The initializer expression must always be assigned a value.
      // Replace this with an assert when we have sufficient coverage of
      // language features.
      if (Value *Val = Env.createValue(D.getType()))
        Env.setValue(Loc, *Val);
    } else {
      llvm_unreachable("structs and classes must always be assigned values");
    }
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *S) {
    // The CFG does not contain `ParenExpr` as top-level statements in basic
    // blocks, however sub-expressions can still be of that type.
    assert(S->getSubExpr() != nullptr);
    const Expr *SubExpr = S->getSubExpr()->IgnoreParens();
    assert(SubExpr != nullptr);

    switch (S->getCastKind()) {
    case CK_LValueToRValue: {
      auto *SubExprVal = Env.getValue(*SubExpr, SkipPast::Reference);
      if (SubExprVal == nullptr)
        break;

      auto &ExprLoc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, ExprLoc);
      Env.setValue(ExprLoc, *SubExprVal);
      break;
    }
    case CK_NoOp: {
      // FIXME: Consider making `Environment::getStorageLocation` skip noop
      // expressions (this and other similar expressions in the file) instead of
      // assigning them storage locations.
      auto *SubExprLoc = Env.getStorageLocation(*SubExpr, SkipPast::None);
      if (SubExprLoc == nullptr)
        break;

      Env.setStorageLocation(*S, *SubExprLoc);
      break;
    }
    default:
      // FIXME: Add support for CK_UserDefinedConversion,
      // CK_ConstructorConversion, CK_UncheckedDerivedToBase.
      break;
    }
  }

  void VisitUnaryOperator(const UnaryOperator *S) {
    // The CFG does not contain `ParenExpr` as top-level statements in basic
    // blocks, however sub-expressions can still be of that type.
    assert(S->getSubExpr() != nullptr);
    const Expr *SubExpr = S->getSubExpr()->IgnoreParens();
    assert(SubExpr != nullptr);

    switch (S->getOpcode()) {
    case UO_Deref: {
      // Skip past a reference to handle dereference of a dependent pointer.
      const auto *SubExprVal = cast_or_null<PointerValue>(
          Env.getValue(*SubExpr, SkipPast::Reference));
      if (SubExprVal == nullptr)
        break;

      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      Env.setValue(Loc, Env.takeOwnership(std::make_unique<ReferenceValue>(
                            SubExprVal->getPointeeLoc())));
      break;
    }
    case UO_AddrOf: {
      // Do not form a pointer to a reference. If `SubExpr` is assigned a
      // `ReferenceValue` then form a value that points to the location of its
      // pointee.
      StorageLocation *PointeeLoc =
          Env.getStorageLocation(*SubExpr, SkipPast::Reference);
      if (PointeeLoc == nullptr)
        break;

      auto &PointerLoc = Env.createStorageLocation(*S);
      auto &PointerVal =
          Env.takeOwnership(std::make_unique<PointerValue>(*PointeeLoc));
      Env.setStorageLocation(*S, PointerLoc);
      Env.setValue(PointerLoc, PointerVal);
      break;
    }
    case UO_LNot: {
      auto *SubExprVal =
          dyn_cast_or_null<BoolValue>(Env.getValue(*SubExpr, SkipPast::None));
      if (SubExprVal == nullptr)
        return;

      auto &ExprLoc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, ExprLoc);
      Env.setValue(ExprLoc, Env.takeOwnership(
                                std::make_unique<NegationValue>(*SubExprVal)));
    }
    default:
      break;
    }
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

  void VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *S) {
    const Expr *InitExpr = S->getExpr();
    assert(InitExpr != nullptr);

    Value *InitExprVal = Env.getValue(*InitExpr, SkipPast::None);
    if (InitExprVal == nullptr)
      return;

    const FieldDecl *Field = S->getField();
    assert(Field != nullptr);

    auto &ThisLoc =
        *cast<AggregateStorageLocation>(Env.getThisPointeeStorageLocation());
    auto &FieldLoc = ThisLoc.getChild(*Field);
    Env.setValue(FieldLoc, *InitExprVal);
  }

  void VisitCXXConstructExpr(const CXXConstructExpr *S) {
    const CXXConstructorDecl *ConstructorDecl = S->getConstructor();
    assert(ConstructorDecl != nullptr);

    if (ConstructorDecl->isCopyOrMoveConstructor()) {
      assert(S->getNumArgs() == 1);

      const Expr *Arg = S->getArg(0);
      assert(Arg != nullptr);

      if (S->isElidable()) {
        auto *ArgLoc = Env.getStorageLocation(*Arg, SkipPast::Reference);
        if (ArgLoc == nullptr)
          return;

        Env.setStorageLocation(*S, *ArgLoc);
      } else if (auto *ArgVal = Env.getValue(*Arg, SkipPast::Reference)) {
        auto &Loc = Env.createStorageLocation(*S);
        Env.setStorageLocation(*S, Loc);
        Env.setValue(Loc, *ArgVal);
      }
      return;
    }

    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(Loc, *Val);
  }

  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *S) {
    if (S->getOperator() == OO_Equal) {
      assert(S->getNumArgs() == 2);

      const Expr *Arg0 = S->getArg(0);
      assert(Arg0 != nullptr);

      const Expr *Arg1 = S->getArg(1);
      assert(Arg1 != nullptr);

      // Evaluate only copy and move assignment operators.
      auto *Arg0Type = Arg0->getType()->getUnqualifiedDesugaredType();
      auto *Arg1Type = Arg1->getType()->getUnqualifiedDesugaredType();
      if (Arg0Type != Arg1Type)
        return;

      auto *ObjectLoc = Env.getStorageLocation(*Arg0, SkipPast::Reference);
      if (ObjectLoc == nullptr)
        return;

      auto *Val = Env.getValue(*Arg1, SkipPast::Reference);
      if (Val == nullptr)
        return;

      Env.setValue(*ObjectLoc, *Val);
    }
  }

  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *S) {
    if (S->getCastKind() == CK_ConstructorConversion) {
      // The CFG does not contain `ParenExpr` as top-level statements in basic
      // blocks, however sub-expressions can still be of that type.
      assert(S->getSubExpr() != nullptr);
      const Expr *SubExpr = S->getSubExpr();
      assert(SubExpr != nullptr);

      auto *SubExprLoc = Env.getStorageLocation(*SubExpr, SkipPast::None);
      if (SubExprLoc == nullptr)
        return;

      Env.setStorageLocation(*S, *SubExprLoc);
    }
  }

  void VisitCXXTemporaryObjectExpr(const CXXTemporaryObjectExpr *S) {
    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(Loc, *Val);
  }

  void VisitCallExpr(const CallExpr *S) {
    if (S->isCallToStdMove()) {
      assert(S->getNumArgs() == 1);

      const Expr *Arg = S->getArg(0);
      assert(Arg != nullptr);

      auto *ArgLoc = Env.getStorageLocation(*Arg, SkipPast::None);
      if (ArgLoc == nullptr)
        return;

      Env.setStorageLocation(*S, *ArgLoc);
    }
  }

  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *S) {
    const Expr *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);

    auto *SubExprLoc = Env.getStorageLocation(*SubExpr, SkipPast::None);
    if (SubExprLoc == nullptr)
      return;

    Env.setStorageLocation(*S, *SubExprLoc);
  }

  void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *S) {
    const Expr *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);

    auto *SubExprLoc = Env.getStorageLocation(*SubExpr, SkipPast::None);
    if (SubExprLoc == nullptr)
      return;

    Env.setStorageLocation(*S, *SubExprLoc);
  }

  void VisitCXXStaticCastExpr(const CXXStaticCastExpr *S) {
    if (S->getCastKind() == CK_NoOp) {
      const Expr *SubExpr = S->getSubExpr();
      assert(SubExpr != nullptr);

      auto *SubExprLoc = Env.getStorageLocation(*SubExpr, SkipPast::None);
      if (SubExprLoc == nullptr)
        return;

      Env.setStorageLocation(*S, *SubExprLoc);
    }
  }

  void VisitConditionalOperator(const ConditionalOperator *S) {
    // FIXME: Revisit this once flow conditions are added to the framework. For
    // `a = b ? c : d` we can add `b => a == c && !b => a == d` to the flow
    // condition.
    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(Loc, *Val);
  }

  void VisitInitListExpr(const InitListExpr *S) {
    QualType Type = S->getType();

    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);

    auto *Val = Env.createValue(Type);
    if (Val == nullptr)
      return;

    Env.setValue(Loc, *Val);

    if (Type->isStructureOrClassType()) {
      for (auto IT : llvm::zip(Type->getAsRecordDecl()->fields(), S->inits())) {
        const FieldDecl *Field = std::get<0>(IT);
        assert(Field != nullptr);

        const Expr *Init = std::get<1>(IT);
        assert(Init != nullptr);

        if (Value *InitVal = Env.getValue(*Init, SkipPast::None))
          cast<StructValue>(Val)->setChild(*Field, *InitVal);
      }
    }
    // FIXME: Implement array initialization.
  }

  void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *S) {
    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    Env.setValue(Loc, Env.getBoolLiteralValue(S->getValue()));
  }

private:
  const StmtToEnvMap &StmtToEnv;
  Environment &Env;
};

void transfer(const StmtToEnvMap &StmtToEnv, const Stmt &S, Environment &Env) {
  assert(!isa<ParenExpr>(&S));
  TransferVisitor(StmtToEnv, Env).Visit(&S);
}

} // namespace dataflow
} // namespace clang
