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
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <memory>
#include <tuple>

namespace clang {
namespace dataflow {

const Environment *StmtToEnvMap::getEnvironment(const Stmt &S) const {
  auto BlockIt = CFCtx.getStmtToBlock().find(&ignoreCFGOmittedNodes(S));
  assert(BlockIt != CFCtx.getStmtToBlock().end());
  if (!CFCtx.isBlockReachable(*BlockIt->getSecond()))
    return nullptr;
  const auto &State = BlockToState[BlockIt->getSecond()->getBlockID()];
  assert(State);
  return &State->Env;
}

static BoolValue &evaluateBooleanEquality(const Expr &LHS, const Expr &RHS,
                                          Environment &Env) {
  if (auto *LHSValue =
          dyn_cast_or_null<BoolValue>(Env.getValue(LHS, SkipPast::Reference)))
    if (auto *RHSValue =
            dyn_cast_or_null<BoolValue>(Env.getValue(RHS, SkipPast::Reference)))
      return Env.makeIff(*LHSValue, *RHSValue);

  return Env.makeAtomicBoolValue();
}

// Functionally updates `V` such that any instances of `TopBool` are replaced
// with fresh atomic bools. Note: This implementation assumes that `B` is a
// tree; if `B` is a DAG, it will lose any sharing between subvalues that was
// present in the original .
static BoolValue &unpackValue(BoolValue &V, Environment &Env);

template <typename Derived, typename M>
BoolValue &unpackBinaryBoolValue(Environment &Env, BoolValue &B, M build) {
  auto &V = *cast<Derived>(&B);
  BoolValue &Left = V.getLeftSubValue();
  BoolValue &Right = V.getRightSubValue();
  BoolValue &ULeft = unpackValue(Left, Env);
  BoolValue &URight = unpackValue(Right, Env);

  if (&ULeft == &Left && &URight == &Right)
    return V;

  return (Env.*build)(ULeft, URight);
}

static BoolValue &unpackValue(BoolValue &V, Environment &Env) {
  switch (V.getKind()) {
  case Value::Kind::Integer:
  case Value::Kind::Reference:
  case Value::Kind::Pointer:
  case Value::Kind::Struct:
    llvm_unreachable("BoolValue cannot have any of these kinds.");

  case Value::Kind::AtomicBool:
    return V;

  case Value::Kind::TopBool:
    // Unpack `TopBool` into a fresh atomic bool.
    return Env.makeAtomicBoolValue();

  case Value::Kind::Negation: {
    auto &N = *cast<NegationValue>(&V);
    BoolValue &Sub = N.getSubVal();
    BoolValue &USub = unpackValue(Sub, Env);

    if (&USub == &Sub)
      return V;
    return Env.makeNot(USub);
  }
  case Value::Kind::Conjunction:
    return unpackBinaryBoolValue<ConjunctionValue>(Env, V,
                                                   &Environment::makeAnd);
  case Value::Kind::Disjunction:
    return unpackBinaryBoolValue<DisjunctionValue>(Env, V,
                                                   &Environment::makeOr);
  case Value::Kind::Implication:
    return unpackBinaryBoolValue<ImplicationValue>(
        Env, V, &Environment::makeImplication);
  case Value::Kind::Biconditional:
    return unpackBinaryBoolValue<BiconditionalValue>(Env, V,
                                                     &Environment::makeIff);
  }
  llvm_unreachable("All reachable cases in switch return");
}

// Unpacks the value (if any) associated with `E` and updates `E` to the new
// value, if any unpacking occured. Also, does the lvalue-to-rvalue conversion,
// by skipping past the reference.
static Value *maybeUnpackLValueExpr(const Expr &E, Environment &Env) {
  // FIXME: this is too flexible: it _allows_ a reference, while it should
  // _require_ one, since lvalues should always be wrapped in `ReferenceValue`.
  auto *Loc = Env.getStorageLocation(E, SkipPast::Reference);
  if (Loc == nullptr)
    return nullptr;
  auto *Val = Env.getValue(*Loc);

  auto *B = dyn_cast_or_null<BoolValue>(Val);
  if (B == nullptr)
    return Val;

  auto &UnpackedVal = unpackValue(*B, Env);
  if (&UnpackedVal == Val)
    return Val;
  Env.setValue(*Loc, UnpackedVal);
  return &UnpackedVal;
}

namespace {

class TransferVisitor : public ConstStmtVisitor<TransferVisitor> {
public:
  TransferVisitor(const StmtToEnvMap &StmtToEnv, Environment &Env)
      : StmtToEnv(StmtToEnv), Env(Env) {}

  void VisitBinaryOperator(const BinaryOperator *S) {
    const Expr *LHS = S->getLHS();
    assert(LHS != nullptr);

    const Expr *RHS = S->getRHS();
    assert(RHS != nullptr);

    switch (S->getOpcode()) {
    case BO_Assign: {
      auto *LHSLoc = Env.getStorageLocation(*LHS, SkipPast::Reference);
      if (LHSLoc == nullptr)
        break;

      // No skipping should be necessary, because any lvalues should have
      // already been stripped off in evaluating the LValueToRValue cast.
      auto *RHSVal = Env.getValue(*RHS, SkipPast::None);
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
      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);

      BoolValue *LHSVal = getLogicOperatorSubExprValue(*LHS);
      // If the LHS was not reachable, this BinaryOperator would also not be
      // reachable, and we would never get here.
      assert(LHSVal != nullptr);
      BoolValue *RHSVal = getLogicOperatorSubExprValue(*RHS);
      if (RHSVal == nullptr) {
        // If the RHS isn't reachable and we evaluate this BinaryOperator,
        // then the value of the LHS must have triggered the short-circuit
        // logic. This implies that the value of the entire expression must be
        // equal to the value of the LHS.
        Env.setValue(Loc, *LHSVal);
        break;
      }

      if (S->getOpcode() == BO_LAnd)
        Env.setValue(Loc, Env.makeAnd(*LHSVal, *RHSVal));
      else
        Env.setValue(Loc, Env.makeOr(*LHSVal, *RHSVal));
      break;
    }
    case BO_NE:
    case BO_EQ: {
      auto &LHSEqRHSValue = evaluateBooleanEquality(*LHS, *RHS, Env);
      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      Env.setValue(Loc, S->getOpcode() == BO_EQ ? LHSEqRHSValue
                                                : Env.makeNot(LHSEqRHSValue));
      break;
    }
    case BO_Comma: {
      if (auto *Loc = Env.getStorageLocation(*RHS, SkipPast::None))
        Env.setStorageLocation(*S, *Loc);
      break;
    }
    default:
      break;
    }
  }

  void VisitDeclRefExpr(const DeclRefExpr *S) {
    const ValueDecl *VD = S->getDecl();
    assert(VD != nullptr);
    auto *DeclLoc = Env.getStorageLocation(*VD, SkipPast::None);
    if (DeclLoc == nullptr)
      return;

    // If the value is already an lvalue, don't double-wrap it.
    if (isa_and_nonnull<ReferenceValue>(Env.getValue(*DeclLoc))) {
      // We only expect to encounter a `ReferenceValue` for a reference type
      // (always) or for `BindingDecl` (sometimes). For the latter, we can't
      // rely on type, because their type does not indicate whether they are a
      // reference type. The assert is not strictly necessary, since we don't
      // depend on its truth to proceed. But, it verifies our assumptions,
      // which, if violated, probably indicate a problem elsewhere.
      assert((VD->getType()->isReferenceType() || isa<BindingDecl>(VD)) &&
             "Only reference-typed declarations or `BindingDecl`s should map "
             "to `ReferenceValue`s");
      Env.setStorageLocation(*S, *DeclLoc);
    } else {
      auto &Loc = Env.createStorageLocation(*S);
      auto &Val = Env.create<ReferenceValue>(*DeclLoc);
      Env.setStorageLocation(*S, Loc);
      Env.setValue(Loc, Val);
    }
  }

  void VisitDeclStmt(const DeclStmt *S) {
    // Group decls are converted into single decls in the CFG so the cast below
    // is safe.
    const auto &D = *cast<VarDecl>(S->getSingleDecl());

    // Static local vars are already initialized in `Environment`.
    if (D.hasGlobalStorage())
      return;

    // The storage location for `D` could have been created earlier, before the
    // variable's declaration statement (for example, in the case of
    // BindingDecls).
    auto *MaybeLoc = Env.getStorageLocation(D, SkipPast::None);
    if (MaybeLoc == nullptr) {
      MaybeLoc = &Env.createStorageLocation(D);
      Env.setStorageLocation(D, *MaybeLoc);
    }
    auto &Loc = *MaybeLoc;

    const Expr *InitExpr = D.getInit();
    if (InitExpr == nullptr) {
      // No initializer expression - associate `Loc` with a new value.
      if (Value *Val = Env.createValue(D.getType()))
        Env.setValue(Loc, *Val);
      return;
    }

    if (D.getType()->isReferenceType()) {
      // Initializing a reference variable - do not create a reference to
      // reference.
      // FIXME: reuse the ReferenceValue instead of creating a new one.
      if (auto *InitExprLoc =
              Env.getStorageLocation(*InitExpr, SkipPast::Reference)) {
        auto &Val = Env.create<ReferenceValue>(*InitExprLoc);
        Env.setValue(Loc, Val);
      }
    } else if (auto *InitExprVal = Env.getValue(*InitExpr, SkipPast::None)) {
      Env.setValue(Loc, *InitExprVal);
    }

    if (Env.getValue(Loc) == nullptr) {
      // We arrive here in (the few) cases where an expression is intentionally
      // "uninterpreted". There are two ways to handle this situation: propagate
      // the status, so that uninterpreted initializers result in uninterpreted
      // variables, or provide a default value. We choose the latter so that
      // later refinements of the variable can be used for reasoning about the
      // surrounding code.
      //
      // FIXME. If and when we interpret all language cases, change this to
      // assert that `InitExpr` is interpreted, rather than supplying a default
      // value (assuming we don't update the environment API to return
      // references).
      if (Value *Val = Env.createValue(D.getType()))
        Env.setValue(Loc, *Val);
    }

    // `DecompositionDecl` must be handled after we've interpreted the loc
    // itself, because the binding expression refers back to the
    // `DecompositionDecl` (even though it has no written name).
    if (const auto *Decomp = dyn_cast<DecompositionDecl>(&D)) {
      // If VarDecl is a DecompositionDecl, evaluate each of its bindings. This
      // needs to be evaluated after initializing the values in the storage for
      // VarDecl, as the bindings refer to them.
      // FIXME: Add support for ArraySubscriptExpr.
      // FIXME: Consider adding AST nodes used in BindingDecls to the CFG.
      for (const auto *B : Decomp->bindings()) {
        if (auto *ME = dyn_cast_or_null<MemberExpr>(B->getBinding())) {
          auto *DE = dyn_cast_or_null<DeclRefExpr>(ME->getBase());
          if (DE == nullptr)
            continue;

          // ME and its base haven't been visited because they aren't included
          // in the statements of the CFG basic block.
          VisitDeclRefExpr(DE);
          VisitMemberExpr(ME);

          if (auto *Loc = Env.getStorageLocation(*ME, SkipPast::Reference))
            Env.setStorageLocation(*B, *Loc);
        } else if (auto *VD = B->getHoldingVar()) {
          // Holding vars are used to back the BindingDecls of tuple-like
          // types. The holding var declarations appear *after* this statement,
          // so we have to create a location for them here to share with `B`. We
          // don't visit the binding, because we know it will be a DeclRefExpr
          // to `VD`. Note that, by construction of the AST, `VD` will always be
          // a reference -- either lvalue or rvalue.
          auto &VDLoc = Env.createStorageLocation(*VD);
          Env.setStorageLocation(*VD, VDLoc);
          Env.setStorageLocation(*B, VDLoc);
        }
      }
    }
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *S) {
    const Expr *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);

    switch (S->getCastKind()) {
    case CK_IntegralToBoolean: {
      // This cast creates a new, boolean value from the integral value. We
      // model that with a fresh value in the environment, unless it's already a
      // boolean.
      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      if (auto *SubExprVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*SubExpr, SkipPast::Reference)))
        Env.setValue(Loc, *SubExprVal);
      else
        // FIXME: If integer modeling is added, then update this code to create
        // the boolean based on the integer model.
        Env.setValue(Loc, Env.makeAtomicBoolValue());
      break;
    }

    case CK_LValueToRValue: {
      // When an L-value is used as an R-value, it may result in sharing, so we
      // need to unpack any nested `Top`s. We also need to strip off the
      // `ReferenceValue` associated with the lvalue.
      auto *SubExprVal = maybeUnpackLValueExpr(*SubExpr, Env);
      if (SubExprVal == nullptr)
        break;

      auto &ExprLoc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, ExprLoc);
      Env.setValue(ExprLoc, *SubExprVal);
      break;
    }

    case CK_IntegralCast:
      // FIXME: This cast creates a new integral value from the
      // subexpression. But, because we don't model integers, we don't
      // distinguish between this new value and the underlying one. If integer
      // modeling is added, then update this code to create a fresh location and
      // value.
    case CK_UncheckedDerivedToBase:
    case CK_ConstructorConversion:
    case CK_UserDefinedConversion:
      // FIXME: Add tests that excercise CK_UncheckedDerivedToBase,
      // CK_ConstructorConversion, and CK_UserDefinedConversion.
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
    case CK_NullToPointer:
    case CK_NullToMemberPointer: {
      auto &Loc = Env.createStorageLocation(S->getType());
      Env.setStorageLocation(*S, Loc);

      auto &NullPointerVal =
          Env.getOrCreateNullPointerValue(S->getType()->getPointeeType());
      Env.setValue(Loc, NullPointerVal);
      break;
    }
    case CK_FunctionToPointerDecay: {
      StorageLocation *PointeeLoc =
          Env.getStorageLocation(*SubExpr, SkipPast::Reference);
      if (PointeeLoc == nullptr)
        break;

      auto &PointerLoc = Env.createStorageLocation(*S);
      auto &PointerVal = Env.create<PointerValue>(*PointeeLoc);
      Env.setStorageLocation(*S, PointerLoc);
      Env.setValue(PointerLoc, PointerVal);
      break;
    }
    default:
      break;
    }
  }

  void VisitUnaryOperator(const UnaryOperator *S) {
    const Expr *SubExpr = S->getSubExpr();
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
      Env.setValue(Loc,
                   Env.create<ReferenceValue>(SubExprVal->getPointeeLoc()));
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
      auto &PointerVal = Env.create<PointerValue>(*PointeeLoc);
      Env.setStorageLocation(*S, PointerLoc);
      Env.setValue(PointerLoc, PointerVal);
      break;
    }
    case UO_LNot: {
      auto *SubExprVal =
          dyn_cast_or_null<BoolValue>(Env.getValue(*SubExpr, SkipPast::None));
      if (SubExprVal == nullptr)
        break;

      auto &ExprLoc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, ExprLoc);
      Env.setValue(ExprLoc, Env.makeNot(*SubExprVal));
      break;
    }
    default:
      break;
    }
  }

  void VisitCXXThisExpr(const CXXThisExpr *S) {
    auto *ThisPointeeLoc = Env.getThisPointeeStorageLocation();
    if (ThisPointeeLoc == nullptr)
      // Unions are not supported yet, and will not have a location for the
      // `this` expression's pointee.
      return;

    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    Env.setValue(Loc, Env.create<PointerValue>(*ThisPointeeLoc));
  }

  void VisitCXXNewExpr(const CXXNewExpr *S) {
    auto &Loc = Env.createStorageLocation(*S);
    Env.setStorageLocation(*S, Loc);
    if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(Loc, *Val);
  }

  void VisitCXXDeleteExpr(const CXXDeleteExpr *S) {
    // Empty method.
    // We consciously don't do anything on deletes.  Diagnosing double deletes
    // (for example) should be done by a specific analysis, not by the
    // framework.
  }

  void VisitReturnStmt(const ReturnStmt *S) {
    if (!Env.getAnalysisOptions().ContextSensitiveOpts)
      return;

    auto *Ret = S->getRetValue();
    if (Ret == nullptr)
      return;

    auto *Val = Env.getValue(*Ret, SkipPast::None);
    if (Val == nullptr)
      return;

    // FIXME: Support reference-type returns.
    if (Val->getKind() == Value::Kind::Reference)
      return;

    auto *Loc = Env.getReturnStorageLocation();
    assert(Loc != nullptr);
    // FIXME: Support reference-type returns.
    if (Loc->getType()->isReferenceType())
      return;

    // FIXME: Model NRVO.
    Env.setValue(*Loc, *Val);
  }

  void VisitMemberExpr(const MemberExpr *S) {
    ValueDecl *Member = S->getMemberDecl();
    assert(Member != nullptr);

    // FIXME: Consider assigning pointer values to function member expressions.
    if (Member->isFunctionOrFunctionTemplate())
      return;

    // FIXME: if/when we add support for modeling enums, use that support here.
    if (isa<EnumConstantDecl>(Member))
      return;

    if (auto *D = dyn_cast<VarDecl>(Member)) {
      if (D->hasGlobalStorage()) {
        auto *VarDeclLoc = Env.getStorageLocation(*D, SkipPast::None);
        if (VarDeclLoc == nullptr)
          return;

        if (VarDeclLoc->getType()->isReferenceType()) {
          assert(isa_and_nonnull<ReferenceValue>(Env.getValue((*VarDeclLoc))) &&
                 "reference-typed declarations map to `ReferenceValue`s");
          Env.setStorageLocation(*S, *VarDeclLoc);
        } else {
          auto &Loc = Env.createStorageLocation(*S);
          Env.setStorageLocation(*S, Loc);
          Env.setValue(Loc, Env.create<ReferenceValue>(*VarDeclLoc));
        }
        return;
      }
    }

    // The receiver can be either a value or a pointer to a value. Skip past the
    // indirection to handle both cases.
    auto *BaseLoc = cast_or_null<AggregateStorageLocation>(
        Env.getStorageLocation(*S->getBase(), SkipPast::ReferenceThenPointer));
    if (BaseLoc == nullptr)
      return;

    auto &MemberLoc = BaseLoc->getChild(*Member);
    if (MemberLoc.getType()->isReferenceType()) {
      // Based on its type, `MemberLoc` must be mapped either to nothing or to a
      // `ReferenceValue`. For the former, we won't set a storage location for
      // this expression, so as to maintain an invariant lvalue expressions;
      // namely, that their location maps to a `ReferenceValue`.  In this,
      // lvalues are unlike other expressions, where it is valid for their
      // location to map to nothing (because they are not modeled).
      //
      // Note: we need this invariant for lvalues so that, when accessing a
      // value, we can distinguish an rvalue from an lvalue. An alternative
      // design, which takes the expression's value category into account, would
      // avoid the need for this invariant.
      if (auto *V = Env.getValue(MemberLoc)) {
        assert(isa<ReferenceValue>(V) &&
               "reference-typed declarations map to `ReferenceValue`s");
        Env.setStorageLocation(*S, MemberLoc);
      }
    } else {
      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      Env.setValue(Loc, Env.create<ReferenceValue>(MemberLoc));
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
      // It is permissible for a copy/move constructor to have additional
      // parameters as long as they have default arguments defined for them.
      assert(S->getNumArgs() != 0);

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

    transferInlineCall(S, ConstructorDecl);
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

      // Assign a value to the storage location of the object.
      Env.setValue(*ObjectLoc, *Val);

      // FIXME: Add a test for the value of the whole expression.
      // Assign a storage location for the whole expression.
      Env.setStorageLocation(*S, *ObjectLoc);
    }
  }

  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *S) {
    if (S->getCastKind() == CK_ConstructorConversion) {
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
    // Of clang's builtins, only `__builtin_expect` is handled explicitly, since
    // others (like trap, debugtrap, and unreachable) are handled by CFG
    // construction.
    if (S->isCallToStdMove()) {
      assert(S->getNumArgs() == 1);

      const Expr *Arg = S->getArg(0);
      assert(Arg != nullptr);

      auto *ArgLoc = Env.getStorageLocation(*Arg, SkipPast::None);
      if (ArgLoc == nullptr)
        return;

      Env.setStorageLocation(*S, *ArgLoc);
    } else if (S->getDirectCallee() != nullptr &&
               S->getDirectCallee()->getBuiltinID() ==
                   Builtin::BI__builtin_expect) {
      assert(S->getNumArgs() > 0);
      assert(S->getArg(0) != nullptr);
      // `__builtin_expect` returns by-value, so strip away any potential
      // references in the argument.
      auto *ArgLoc = Env.getStorageLocation(*S->getArg(0), SkipPast::Reference);
      if (ArgLoc == nullptr)
        return;
      Env.setStorageLocation(*S, *ArgLoc);
    } else if (const FunctionDecl *F = S->getDirectCallee()) {
      transferInlineCall(S, F);
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
      // Unnamed bitfields are only used for padding and are not appearing in
      // `InitListExpr`'s inits. However, those fields do appear in RecordDecl's
      // field list, and we thus need to remove them before mapping inits to
      // fields to avoid mapping inits to the wrongs fields.
      std::vector<FieldDecl *> Fields;
      llvm::copy_if(
          Type->getAsRecordDecl()->fields(), std::back_inserter(Fields),
          [](const FieldDecl *Field) { return !Field->isUnnamedBitfield(); });
      for (auto It : llvm::zip(Fields, S->inits())) {
        const FieldDecl *Field = std::get<0>(It);
        assert(Field != nullptr);

        const Expr *Init = std::get<1>(It);
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

  void VisitParenExpr(const ParenExpr *S) {
    // The CFG does not contain `ParenExpr` as top-level statements in basic
    // blocks, however manual traversal to sub-expressions may encounter them.
    // Redirect to the sub-expression.
    auto *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);
    Visit(SubExpr);
  }

  void VisitExprWithCleanups(const ExprWithCleanups *S) {
    // The CFG does not contain `ExprWithCleanups` as top-level statements in
    // basic blocks, however manual traversal to sub-expressions may encounter
    // them. Redirect to the sub-expression.
    auto *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);
    Visit(SubExpr);
  }

private:
  /// If `SubExpr` is reachable, returns a non-null pointer to the value for
  /// `SubExpr`. If `SubExpr` is not reachable, returns nullptr.
  BoolValue *getLogicOperatorSubExprValue(const Expr &SubExpr) {
    // `SubExpr` and its parent logic operator might be part of different basic
    // blocks. We try to access the value that is assigned to `SubExpr` in the
    // corresponding environment.
    const Environment *SubExprEnv = StmtToEnv.getEnvironment(SubExpr);
    if (!SubExprEnv)
      return nullptr;

    if (auto *Val = dyn_cast_or_null<BoolValue>(
            SubExprEnv->getValue(SubExpr, SkipPast::Reference)))
      return Val;

    if (Env.getStorageLocation(SubExpr, SkipPast::None) == nullptr) {
      // Sub-expressions that are logic operators are not added in basic blocks
      // (e.g. see CFG for `bool d = a && (b || c);`). If `SubExpr` is a logic
      // operator, it may not have been evaluated and assigned a value yet. In
      // that case, we need to first visit `SubExpr` and then try to get the
      // value that gets assigned to it.
      Visit(&SubExpr);
    }

    if (auto *Val = dyn_cast_or_null<BoolValue>(
            Env.getValue(SubExpr, SkipPast::Reference)))
      return Val;

    // If the value of `SubExpr` is still unknown, we create a fresh symbolic
    // boolean value for it.
    return &Env.makeAtomicBoolValue();
  }

  // If context sensitivity is enabled, try to analyze the body of the callee
  // `F` of `S`. The type `E` must be either `CallExpr` or `CXXConstructExpr`.
  template <typename E>
  void transferInlineCall(const E *S, const FunctionDecl *F) {
    const auto &Options = Env.getAnalysisOptions();
    if (!(Options.ContextSensitiveOpts &&
          Env.canDescend(Options.ContextSensitiveOpts->Depth, F)))
      return;

    const ControlFlowContext *CFCtx = Env.getControlFlowContext(F);
    if (!CFCtx)
      return;

    // FIXME: We don't support context-sensitive analysis of recursion, so
    // we should return early here if `F` is the same as the `FunctionDecl`
    // holding `S` itself.

    auto ExitBlock = CFCtx->getCFG().getExit().getBlockID();

    if (const auto *NonConstructExpr = dyn_cast<CallExpr>(S)) {
      // Note that it is important for the storage location of `S` to be set
      // before `pushCall`, because the latter uses it to set the storage
      // location for `return`.
      auto &ReturnLoc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, ReturnLoc);
    }
    auto CalleeEnv = Env.pushCall(S);

    // FIXME: Use the same analysis as the caller for the callee. Note,
    // though, that doing so would require support for changing the analysis's
    // ASTContext.
    assert(CFCtx->getDecl() != nullptr &&
           "ControlFlowContexts in the environment should always carry a decl");
    auto Analysis = NoopAnalysis(CFCtx->getDecl()->getASTContext(),
                                 DataflowAnalysisOptions{Options});

    auto BlockToOutputState =
        dataflow::runDataflowAnalysis(*CFCtx, Analysis, CalleeEnv);
    assert(BlockToOutputState);
    assert(ExitBlock < BlockToOutputState->size());

    auto ExitState = (*BlockToOutputState)[ExitBlock];
    assert(ExitState);

    Env.popCall(ExitState->Env);
  }

  const StmtToEnvMap &StmtToEnv;
  Environment &Env;
};

} // namespace

void transfer(const StmtToEnvMap &StmtToEnv, const Stmt &S, Environment &Env) {
  TransferVisitor(StmtToEnv, Env).Visit(&S);
}

} // namespace dataflow
} // namespace clang
