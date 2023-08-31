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
#include "clang/Analysis/FlowSensitive/RecordOps.h"
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
  Value *LHSValue = Env.getValue(LHS);
  Value *RHSValue = Env.getValue(RHS);

  if (LHSValue == RHSValue)
    return Env.getBoolLiteralValue(true);

  if (auto *LHSBool = dyn_cast_or_null<BoolValue>(LHSValue))
    if (auto *RHSBool = dyn_cast_or_null<BoolValue>(RHSValue))
      return Env.makeIff(*LHSBool, *RHSBool);

  return Env.makeAtomicBoolValue();
}

static BoolValue &unpackValue(BoolValue &V, Environment &Env) {
  if (auto *Top = llvm::dyn_cast<TopBoolValue>(&V)) {
    auto &A = Env.getDataflowAnalysisContext().arena();
    return A.makeBoolValue(A.makeAtomRef(Top->getAtom()));
  }
  return V;
}

// Unpacks the value (if any) associated with `E` and updates `E` to the new
// value, if any unpacking occured. Also, does the lvalue-to-rvalue conversion,
// by skipping past the reference.
static Value *maybeUnpackLValueExpr(const Expr &E, Environment &Env) {
  auto *Loc = Env.getStorageLocation(E);
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

static void propagateValue(const Expr &From, const Expr &To, Environment &Env) {
  if (auto *Val = Env.getValue(From))
    Env.setValue(To, *Val);
}

static void propagateStorageLocation(const Expr &From, const Expr &To,
                                     Environment &Env) {
  if (auto *Loc = Env.getStorageLocation(From))
    Env.setStorageLocation(To, *Loc);
}

// Propagates the value or storage location of `From` to `To` in cases where
// `From` may be either a glvalue or a prvalue. `To` must be a glvalue iff
// `From` is a glvalue.
static void propagateValueOrStorageLocation(const Expr &From, const Expr &To,
                                            Environment &Env) {
  assert(From.isGLValue() == To.isGLValue());
  if (From.isGLValue())
    propagateStorageLocation(From, To, Env);
  else
    propagateValue(From, To, Env);
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
      auto *LHSLoc = Env.getStorageLocation(*LHS);
      if (LHSLoc == nullptr)
        break;

      auto *RHSVal = Env.getValue(*RHS);
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
      BoolValue &LHSVal = getLogicOperatorSubExprValue(*LHS);
      BoolValue &RHSVal = getLogicOperatorSubExprValue(*RHS);

      if (S->getOpcode() == BO_LAnd)
        Env.setValue(*S, Env.makeAnd(LHSVal, RHSVal));
      else
        Env.setValue(*S, Env.makeOr(LHSVal, RHSVal));
      break;
    }
    case BO_NE:
    case BO_EQ: {
      auto &LHSEqRHSValue = evaluateBooleanEquality(*LHS, *RHS, Env);
      Env.setValue(*S, S->getOpcode() == BO_EQ ? LHSEqRHSValue
                                               : Env.makeNot(LHSEqRHSValue));
      break;
    }
    case BO_Comma: {
      propagateValueOrStorageLocation(*RHS, *S, Env);
      break;
    }
    default:
      break;
    }
  }

  void VisitDeclRefExpr(const DeclRefExpr *S) {
    const ValueDecl *VD = S->getDecl();
    assert(VD != nullptr);

    // Some `DeclRefExpr`s aren't glvalues, so we can't associate them with a
    // `StorageLocation`, and there's also no sensible `Value` that we can
    // assign to them. Examples:
    // - Non-static member variables
    // - Non static member functions
    //   Note: Member operators are an exception to this, but apparently only
    //   if the `DeclRefExpr` is used within the callee of a
    //   `CXXOperatorCallExpr`. In other cases, for example when applying the
    //   address-of operator, the `DeclRefExpr` is a prvalue.
    if (!S->isGLValue())
      return;

    auto *DeclLoc = Env.getStorageLocation(*VD);
    if (DeclLoc == nullptr)
      return;

    Env.setStorageLocation(*S, *DeclLoc);
  }

  void VisitDeclStmt(const DeclStmt *S) {
    // Group decls are converted into single decls in the CFG so the cast below
    // is safe.
    const auto &D = *cast<VarDecl>(S->getSingleDecl());

    ProcessVarDecl(D);
  }

  void ProcessVarDecl(const VarDecl &D) {
    // Static local vars are already initialized in `Environment`.
    if (D.hasGlobalStorage())
      return;

    // If this is the holding variable for a `BindingDecl`, we may already
    // have a storage location set up -- so check. (See also explanation below
    // where we process the `BindingDecl`.)
    if (D.getType()->isReferenceType() && Env.getStorageLocation(D) != nullptr)
      return;

    assert(Env.getStorageLocation(D) == nullptr);

    Env.setStorageLocation(D, Env.createObject(D));

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

          if (auto *Loc = Env.getStorageLocation(*ME))
            Env.setStorageLocation(*B, *Loc);
        } else if (auto *VD = B->getHoldingVar()) {
          // Holding vars are used to back the `BindingDecl`s of tuple-like
          // types. The holding var declarations appear after the
          // `DecompositionDecl`, so we have to explicitly process them here
          // to know their storage location. They will be processed a second
          // time when we visit their `VarDecl`s, so we have code that protects
          // against this above.
          ProcessVarDecl(*VD);
          auto *VDLoc = Env.getStorageLocation(*VD);
          assert(VDLoc != nullptr);
          Env.setStorageLocation(*B, *VDLoc);
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
      if (auto *SubExprVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*SubExpr)))
        Env.setValue(*S, *SubExprVal);
      else
        // FIXME: If integer modeling is added, then update this code to create
        // the boolean based on the integer model.
        Env.setValue(*S, Env.makeAtomicBoolValue());
      break;
    }

    case CK_LValueToRValue: {
      // When an L-value is used as an R-value, it may result in sharing, so we
      // need to unpack any nested `Top`s. We also need to strip off the
      // `ReferenceValue` associated with the lvalue.
      auto *SubExprVal = maybeUnpackLValueExpr(*SubExpr, Env);
      if (SubExprVal == nullptr)
        break;

      Env.setValue(*S, *SubExprVal);
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
      // expressions (this and other similar expressions in the file) instead
      // of assigning them storage locations.
      propagateValueOrStorageLocation(*SubExpr, *S, Env);
      break;
    }
    case CK_NullToPointer: {
      auto &NullPointerVal =
          Env.getOrCreateNullPointerValue(S->getType()->getPointeeType());
      Env.setValue(*S, NullPointerVal);
      break;
    }
    case CK_NullToMemberPointer:
      // FIXME: Implement pointers to members. For now, don't associate a value
      // with this expression.
      break;
    case CK_FunctionToPointerDecay: {
      StorageLocation *PointeeLoc = Env.getStorageLocation(*SubExpr);
      if (PointeeLoc == nullptr)
        break;

      Env.setValue(*S, Env.create<PointerValue>(*PointeeLoc));
      break;
    }
    case CK_BuiltinFnToFnPtr:
      // Despite its name, the result type of `BuiltinFnToFnPtr` is a function,
      // not a function pointer. In addition, builtin functions can only be
      // called directly; it is not legal to take their address. We therefore
      // don't need to create a value or storage location for them.
      break;
    default:
      break;
    }
  }

  void VisitUnaryOperator(const UnaryOperator *S) {
    const Expr *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);

    switch (S->getOpcode()) {
    case UO_Deref: {
      const auto *SubExprVal =
          cast_or_null<PointerValue>(Env.getValue(*SubExpr));
      if (SubExprVal == nullptr)
        break;

      Env.setStorageLocation(*S, SubExprVal->getPointeeLoc());
      break;
    }
    case UO_AddrOf: {
      // FIXME: Model pointers to members.
      if (S->getType()->isMemberPointerType())
        break;

      if (StorageLocation *PointeeLoc = Env.getStorageLocation(*SubExpr))
        Env.setValue(*S, Env.create<PointerValue>(*PointeeLoc));
      break;
    }
    case UO_LNot: {
      auto *SubExprVal = dyn_cast_or_null<BoolValue>(Env.getValue(*SubExpr));
      if (SubExprVal == nullptr)
        break;

      Env.setValue(*S, Env.makeNot(*SubExprVal));
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

    Env.setValue(*S, Env.create<PointerValue>(*ThisPointeeLoc));
  }

  void VisitCXXNewExpr(const CXXNewExpr *S) {
    if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(*S, *Val);
  }

  void VisitCXXDeleteExpr(const CXXDeleteExpr *S) {
    // Empty method.
    // We consciously don't do anything on deletes.  Diagnosing double deletes
    // (for example) should be done by a specific analysis, not by the
    // framework.
  }

  void VisitReturnStmt(const ReturnStmt *S) {
    if (!Env.getDataflowAnalysisContext().getOptions().ContextSensitiveOpts)
      return;

    auto *Ret = S->getRetValue();
    if (Ret == nullptr)
      return;

    if (Ret->isPRValue()) {
      auto *Val = Env.getValue(*Ret);
      if (Val == nullptr)
        return;

      // FIXME: Model NRVO.
      Env.setReturnValue(Val);
    } else {
      auto *Loc = Env.getStorageLocation(*Ret);
      if (Loc == nullptr)
        return;

      // FIXME: Model NRVO.
      Env.setReturnStorageLocation(Loc);
    }
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
        auto *VarDeclLoc = Env.getStorageLocation(*D);
        if (VarDeclLoc == nullptr)
          return;

        Env.setStorageLocation(*S, *VarDeclLoc);
        return;
      }
    }

    RecordStorageLocation *BaseLoc = getBaseObjectLocation(*S, Env);
    if (BaseLoc == nullptr)
      return;

    auto *MemberLoc = BaseLoc->getChild(*Member);
    if (MemberLoc == nullptr)
      return;
    Env.setStorageLocation(*S, *MemberLoc);
  }

  void VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *S) {
    const Expr *InitExpr = S->getExpr();
    assert(InitExpr != nullptr);
    propagateValueOrStorageLocation(*InitExpr, *S, Env);
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

      auto *ArgLoc =
          cast_or_null<RecordStorageLocation>(Env.getStorageLocation(*Arg));
      if (ArgLoc == nullptr)
        return;

      if (S->isElidable()) {
        if (Value *Val = Env.getValue(*ArgLoc))
          Env.setValue(*S, *Val);
      } else {
        auto &Val = *cast<RecordValue>(Env.createValue(S->getType()));
        Env.setValue(*S, Val);
        copyRecord(*ArgLoc, Val.getLoc(), Env);
      }
      return;
    }

    // `CXXConstructExpr` can have array type if default-initializing an array
    // of records, and we currently can't create values for arrays. So check if
    // we've got a record type.
    if (S->getType()->isRecordType()) {
      auto &InitialVal = *cast<RecordValue>(Env.createValue(S->getType()));
      Env.setValue(*S, InitialVal);
      copyRecord(InitialVal.getLoc(), Env.getResultObjectLocation(*S), Env);
    }

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
      const auto *Method =
          dyn_cast_or_null<CXXMethodDecl>(S->getDirectCallee());
      if (!Method)
        return;
      if (!Method->isCopyAssignmentOperator() &&
          !Method->isMoveAssignmentOperator())
        return;

      auto *LocSrc =
          cast_or_null<RecordStorageLocation>(Env.getStorageLocation(*Arg1));
      auto *LocDst =
          cast_or_null<RecordStorageLocation>(Env.getStorageLocation(*Arg0));

      if (LocSrc != nullptr && LocDst != nullptr) {
        copyRecord(*LocSrc, *LocDst, Env);
        Env.setStorageLocation(*S, *LocDst);
      }
    }
  }

  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *S) {
    if (S->getCastKind() == CK_ConstructorConversion) {
      const Expr *SubExpr = S->getSubExpr();
      assert(SubExpr != nullptr);

      propagateValue(*SubExpr, *S, Env);
    }
  }

  void VisitCXXTemporaryObjectExpr(const CXXTemporaryObjectExpr *S) {
    if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(*S, *Val);
  }

  void VisitCallExpr(const CallExpr *S) {
    // Of clang's builtins, only `__builtin_expect` is handled explicitly, since
    // others (like trap, debugtrap, and unreachable) are handled by CFG
    // construction.
    if (S->isCallToStdMove()) {
      assert(S->getNumArgs() == 1);

      const Expr *Arg = S->getArg(0);
      assert(Arg != nullptr);

      auto *ArgLoc = Env.getStorageLocation(*Arg);
      if (ArgLoc == nullptr)
        return;

      Env.setStorageLocation(*S, *ArgLoc);
    } else if (S->getDirectCallee() != nullptr &&
               S->getDirectCallee()->getBuiltinID() ==
                   Builtin::BI__builtin_expect) {
      assert(S->getNumArgs() > 0);
      assert(S->getArg(0) != nullptr);
      auto *ArgVal = Env.getValue(*S->getArg(0));
      if (ArgVal == nullptr)
        return;
      Env.setValue(*S, *ArgVal);
    } else if (const FunctionDecl *F = S->getDirectCallee()) {
      transferInlineCall(S, F);
    }
  }

  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *S) {
    const Expr *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);

    Value *SubExprVal = Env.getValue(*SubExpr);
    if (SubExprVal == nullptr)
      return;

    if (RecordValue *RecordVal = dyn_cast<RecordValue>(SubExprVal)) {
      Env.setStorageLocation(*S, RecordVal->getLoc());
      return;
    }

    StorageLocation &Loc = Env.createStorageLocation(*S);
    Env.setValue(Loc, *SubExprVal);
    Env.setStorageLocation(*S, Loc);
  }

  void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *S) {
    const Expr *SubExpr = S->getSubExpr();
    assert(SubExpr != nullptr);

    propagateValue(*SubExpr, *S, Env);
  }

  void VisitCXXStaticCastExpr(const CXXStaticCastExpr *S) {
    if (S->getCastKind() == CK_NoOp) {
      const Expr *SubExpr = S->getSubExpr();
      assert(SubExpr != nullptr);

      propagateValueOrStorageLocation(*SubExpr, *S, Env);
    }
  }

  void VisitConditionalOperator(const ConditionalOperator *S) {
    // FIXME: Revisit this once flow conditions are added to the framework. For
    // `a = b ? c : d` we can add `b => a == c && !b => a == d` to the flow
    // condition.
    if (S->isGLValue())
      Env.setStorageLocation(*S, Env.createObject(S->getType()));
    else if (Value *Val = Env.createValue(S->getType()))
      Env.setValue(*S, *Val);
  }

  void VisitInitListExpr(const InitListExpr *S) {
    QualType Type = S->getType();

    if (!Type->isStructureOrClassType()) {
      if (auto *Val = Env.createValue(Type))
        Env.setValue(*S, *Val);

      return;
    }

    // In case the initializer list is transparent, we just need to propagate
    // the value that it contains.
    if (S->isSemanticForm() && S->isTransparent()) {
      propagateValue(*S->getInit(0), *S, Env);
      return;
    }

    std::vector<FieldDecl *> Fields =
        getFieldsForInitListExpr(Type->getAsRecordDecl());
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;

    for (auto [Field, Init] : llvm::zip(Fields, S->inits())) {
      assert(Field != nullptr);
      assert(Init != nullptr);

      FieldLocs.insert({Field, &Env.createObject(Field->getType(), Init)});
    }

    auto &Loc =
        Env.getDataflowAnalysisContext().arena().create<RecordStorageLocation>(
            Type, std::move(FieldLocs));
    RecordValue &RecordVal = Env.create<RecordValue>(Loc);

    Env.setValue(Loc, RecordVal);

    Env.setValue(*S, RecordVal);

    // FIXME: Implement array initialization.
  }

  void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *S) {
    Env.setValue(*S, Env.getBoolLiteralValue(S->getValue()));
  }

  void VisitIntegerLiteral(const IntegerLiteral *S) {
    Env.setValue(*S, Env.getIntLiteralValue(S->getValue()));
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
  /// Returns the value for the sub-expression `SubExpr` of a logic operator.
  BoolValue &getLogicOperatorSubExprValue(const Expr &SubExpr) {
    // `SubExpr` and its parent logic operator might be part of different basic
    // blocks. We try to access the value that is assigned to `SubExpr` in the
    // corresponding environment.
    if (const Environment *SubExprEnv = StmtToEnv.getEnvironment(SubExpr))
      if (auto *Val =
              dyn_cast_or_null<BoolValue>(SubExprEnv->getValue(SubExpr)))
        return *Val;

    // The sub-expression may lie within a basic block that isn't reachable,
    // even if we need it to evaluate the current (reachable) expression
    // (see https://discourse.llvm.org/t/70775). In this case, visit `SubExpr`
    // within the current environment and then try to get the value that gets
    // assigned to it.
    if (Env.getValue(SubExpr) == nullptr)
      Visit(&SubExpr);
    if (auto *Val = dyn_cast_or_null<BoolValue>(Env.getValue(SubExpr)))
      return *Val;

    // If the value of `SubExpr` is still unknown, we create a fresh symbolic
    // boolean value for it.
    return Env.makeAtomicBoolValue();
  }

  // If context sensitivity is enabled, try to analyze the body of the callee
  // `F` of `S`. The type `E` must be either `CallExpr` or `CXXConstructExpr`.
  template <typename E>
  void transferInlineCall(const E *S, const FunctionDecl *F) {
    const auto &Options = Env.getDataflowAnalysisContext().getOptions();
    if (!(Options.ContextSensitiveOpts &&
          Env.canDescend(Options.ContextSensitiveOpts->Depth, F)))
      return;

    const ControlFlowContext *CFCtx =
        Env.getDataflowAnalysisContext().getControlFlowContext(F);
    if (!CFCtx)
      return;

    // FIXME: We don't support context-sensitive analysis of recursion, so
    // we should return early here if `F` is the same as the `FunctionDecl`
    // holding `S` itself.

    auto ExitBlock = CFCtx->getCFG().getExit().getBlockID();

    auto CalleeEnv = Env.pushCall(S);

    // FIXME: Use the same analysis as the caller for the callee. Note,
    // though, that doing so would require support for changing the analysis's
    // ASTContext.
    auto Analysis = NoopAnalysis(CFCtx->getDecl().getASTContext(),
                                 DataflowAnalysisOptions{Options});

    auto BlockToOutputState =
        dataflow::runDataflowAnalysis(*CFCtx, Analysis, CalleeEnv);
    assert(BlockToOutputState);
    assert(ExitBlock < BlockToOutputState->size());

    auto &ExitState = (*BlockToOutputState)[ExitBlock];
    assert(ExitState);

    Env.popCall(S, ExitState->Env);
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
