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
#include <cassert>
#include <memory>
#include <tuple>

namespace clang {
namespace dataflow {

static BoolValue &evaluateBooleanEquality(const Expr &LHS, const Expr &RHS,
                                          Environment &Env) {
  if (auto *LHSValue =
          dyn_cast_or_null<BoolValue>(Env.getValue(LHS, SkipPast::Reference)))
    if (auto *RHSValue =
            dyn_cast_or_null<BoolValue>(Env.getValue(RHS, SkipPast::Reference)))
      return Env.makeIff(*LHSValue, *RHSValue);

  return Env.makeAtomicBoolValue();
}

class TransferVisitor : public ConstStmtVisitor<TransferVisitor> {
public:
  TransferVisitor(const StmtToEnvMap &StmtToEnv, Environment &Env,
                  TransferOptions Options)
      : StmtToEnv(StmtToEnv), Env(Env), Options(Options) {}

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

      auto *RHSVal = Env.getValue(*RHS, SkipPast::Reference);
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

      auto &Loc = Env.createStorageLocation(*S);
      Env.setStorageLocation(*S, Loc);
      if (S->getOpcode() == BO_LAnd)
        Env.setValue(Loc, Env.makeAnd(LHSVal, RHSVal));
      else
        Env.setValue(Loc, Env.makeOr(LHSVal, RHSVal));
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

    // Static local vars are already initialized in `Environment`.
    if (D.hasGlobalStorage())
      return;

    auto &Loc = Env.createStorageLocation(D);
    Env.setStorageLocation(D, Loc);

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
      if (auto *InitExprLoc =
              Env.getStorageLocation(*InitExpr, SkipPast::Reference)) {
        auto &Val =
            Env.takeOwnership(std::make_unique<ReferenceValue>(*InitExprLoc));
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

    if (const auto *Decomp = dyn_cast<DecompositionDecl>(&D)) {
      // If VarDecl is a DecompositionDecl, evaluate each of its bindings. This
      // needs to be evaluated after initializing the values in the storage for
      // VarDecl, as the bindings refer to them.
      // FIXME: Add support for ArraySubscriptExpr.
      // FIXME: Consider adding AST nodes that are used for structured bindings
      // to the CFG.
      for (const auto *B : Decomp->bindings()) {
        auto *ME = dyn_cast_or_null<MemberExpr>(B->getBinding());
        if (ME == nullptr)
          continue;

        auto *DE = dyn_cast_or_null<DeclRefExpr>(ME->getBase());
        if (DE == nullptr)
          continue;

        // ME and its base haven't been visited because they aren't included in
        // the statements of the CFG basic block.
        VisitDeclRefExpr(DE);
        VisitMemberExpr(ME);

        if (auto *Loc = Env.getStorageLocation(*ME, SkipPast::Reference))
          Env.setStorageLocation(*B, *Loc);
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
      auto *SubExprVal = Env.getValue(*SubExpr, SkipPast::Reference);
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
    Env.setValue(Loc, Env.takeOwnership(
                          std::make_unique<PointerValue>(*ThisPointeeLoc)));
  }

  void VisitMemberExpr(const MemberExpr *S) {
    ValueDecl *Member = S->getMemberDecl();
    assert(Member != nullptr);

    // FIXME: Consider assigning pointer values to function member expressions.
    if (Member->isFunctionOrFunctionTemplate())
      return;

    if (auto *D = dyn_cast<VarDecl>(Member)) {
      if (D->hasGlobalStorage()) {
        auto *VarDeclLoc = Env.getStorageLocation(*D, SkipPast::None);
        if (VarDeclLoc == nullptr)
          return;

        if (VarDeclLoc->getType()->isReferenceType()) {
          Env.setStorageLocation(*S, *VarDeclLoc);
        } else {
          auto &Loc = Env.createStorageLocation(*S);
          Env.setStorageLocation(*S, Loc);
          Env.setValue(Loc, Env.takeOwnership(
                                std::make_unique<ReferenceValue>(*VarDeclLoc)));
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
      // This case is for context-sensitive analysis, which we only do if we
      // have the callee body available in the translation unit.
      if (!Options.ContextSensitive || F->getBody() == nullptr)
        return;

      // FIXME: We don't support context-sensitive analysis of recursion, so
      // we should return early here if `F` is the same as the `FunctionDecl`
      // holding `S` itself.

      auto &ASTCtx = F->getASTContext();

      // FIXME: Cache these CFGs.
      auto CFCtx = ControlFlowContext::build(F, F->getBody(), &ASTCtx);
      // FIXME: Handle errors here and below.
      assert(CFCtx);
      auto ExitBlock = CFCtx->getCFG().getExit().getBlockID();

      auto CalleeEnv = Env.pushCall(S);

      // FIXME: Use the same analysis as the caller for the callee.
      DataflowAnalysisOptions Options;
      auto Analysis = NoopAnalysis(ASTCtx, Options);

      auto BlockToOutputState =
          dataflow::runDataflowAnalysis(*CFCtx, Analysis, CalleeEnv);
      assert(BlockToOutputState);
      assert(ExitBlock < BlockToOutputState->size());

      auto ExitState = (*BlockToOutputState)[ExitBlock];
      assert(ExitState);

      Env.popCall(ExitState->Env);
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
      for (auto It : llvm::zip(Type->getAsRecordDecl()->fields(), S->inits())) {
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
  BoolValue &getLogicOperatorSubExprValue(const Expr &SubExpr) {
    // `SubExpr` and its parent logic operator might be part of different basic
    // blocks. We try to access the value that is assigned to `SubExpr` in the
    // corresponding environment.
    if (const Environment *SubExprEnv = StmtToEnv.getEnvironment(SubExpr)) {
      if (auto *Val = dyn_cast_or_null<BoolValue>(
              SubExprEnv->getValue(SubExpr, SkipPast::Reference)))
        return *Val;
    }

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
      return *Val;

    // If the value of `SubExpr` is still unknown, we create a fresh symbolic
    // boolean value for it.
    return Env.makeAtomicBoolValue();
  }

  const StmtToEnvMap &StmtToEnv;
  Environment &Env;
  TransferOptions Options;
};

void transfer(const StmtToEnvMap &StmtToEnv, const Stmt &S, Environment &Env,
              TransferOptions Options) {
  TransferVisitor(StmtToEnv, Env, Options).Visit(&S);
}

} // namespace dataflow
} // namespace clang
