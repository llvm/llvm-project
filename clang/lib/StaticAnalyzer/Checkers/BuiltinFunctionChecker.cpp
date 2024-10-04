//=== BuiltinFunctionChecker.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker evaluates "standalone" clang builtin functions that are not
// just special-cased variants of well-known non-builtin functions.
// Builtin functions like __builtin_memcpy and __builtin_alloca should be
// evaluated by the same checker that handles their non-builtin variant to
// ensure that the two variants are handled consistently.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/Taint.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"

using namespace clang;
using namespace ento;
using namespace taint;

namespace {

QualType getSufficientTypeForOverflowOp(CheckerContext &C, const QualType &T) {
  // Calling a builtin with a non-integer type result produces compiler error.
  assert(T->isIntegerType());

  ASTContext &ACtx = C.getASTContext();

  unsigned BitWidth = ACtx.getIntWidth(T);
  return ACtx.getIntTypeForBitwidth(BitWidth * 2, T->isSignedIntegerType());
}

QualType getOverflowBuiltinResultType(const CallEvent &Call) {
  // Calling a builtin with an incorrect argument count produces compiler error.
  assert(Call.getNumArgs() == 3);

  return Call.getArgExpr(2)->getType()->getPointeeType();
}

QualType getOverflowBuiltinResultType(const CallEvent &Call, CheckerContext &C,
                                      unsigned BI) {
  // Calling a builtin with an incorrect argument count produces compiler error.
  assert(Call.getNumArgs() == 3);

  ASTContext &ACtx = C.getASTContext();

  switch (BI) {
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_sadd_overflow:
    return ACtx.IntTy;
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_saddl_overflow:
    return ACtx.LongTy;
  case Builtin::BI__builtin_smulll_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_saddll_overflow:
    return ACtx.LongLongTy;
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_uadd_overflow:
    return ACtx.UnsignedIntTy;
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
    return ACtx.UnsignedLongTy;
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
    return ACtx.UnsignedLongLongTy;
  case Builtin::BI__builtin_mul_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_add_overflow:
    return getOverflowBuiltinResultType(Call);
  default:
    assert(false && "Unknown overflow builtin");
    return ACtx.IntTy;
  }
}

class BuiltinFunctionChecker : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void handleOverflowBuiltin(const CallEvent &Call, CheckerContext &C,
                             BinaryOperator::Opcode Op,
                             QualType ResultType) const;
  const NoteTag *createBuiltinNoOverflowNoteTag(CheckerContext &C,
                                                bool BothFeasible, SVal Arg1,
                                                SVal Arg2, SVal Result) const;
  const NoteTag *createBuiltinOverflowNoteTag(CheckerContext &C) const;
  std::pair<bool, bool> checkOverflow(CheckerContext &C, SVal RetVal,
                                      QualType Res) const;

private:
  // From: clang/include/clang/Basic/Builtins.def
  // C++ standard library builtins in namespace 'std'.
  const CallDescriptionSet BuiltinLikeStdFunctions{
      {CDM::SimpleFunc, {"std", "addressof"}},        //
      {CDM::SimpleFunc, {"std", "__addressof"}},      //
      {CDM::SimpleFunc, {"std", "as_const"}},         //
      {CDM::SimpleFunc, {"std", "forward"}},          //
      {CDM::SimpleFunc, {"std", "forward_like"}},     //
      {CDM::SimpleFunc, {"std", "move"}},             //
      {CDM::SimpleFunc, {"std", "move_if_noexcept"}}, //
  };

  bool isBuiltinLikeFunction(const CallEvent &Call) const;
};

} // namespace

const NoteTag *BuiltinFunctionChecker::createBuiltinNoOverflowNoteTag(
    CheckerContext &C, bool BothFeasible, SVal Arg1, SVal Arg2,
    SVal Result) const {
  return C.getNoteTag([Result, Arg1, Arg2, BothFeasible](
                          PathSensitiveBugReport &BR, llvm::raw_ostream &OS) {
    if (!BR.isInteresting(Result))
      return;

    // Propagate interestingness to input argumets if result is interesting.
    BR.markInteresting(Arg1);
    BR.markInteresting(Arg2);

    if (BothFeasible)
      OS << "Assuming no overflow";
  });
}

const NoteTag *
BuiltinFunctionChecker::createBuiltinOverflowNoteTag(CheckerContext &C) const {
  return C.getNoteTag([](PathSensitiveBugReport &BR,
                         llvm::raw_ostream &OS) { OS << "Assuming overflow"; },
                      /*isPrunable=*/true);
}

std::pair<bool, bool>
BuiltinFunctionChecker::checkOverflow(CheckerContext &C, SVal RetVal,
                                      QualType Res) const {
  // Calling a builtin with a non-integer type result produces compiler error.
  assert(Res->isIntegerType());

  unsigned BitWidth = C.getASTContext().getIntWidth(Res);
  bool IsUnsigned = Res->isUnsignedIntegerType();

  auto MinValType = llvm::APSInt::getMinValue(BitWidth, IsUnsigned);
  auto MaxValType = llvm::APSInt::getMaxValue(BitWidth, IsUnsigned);
  nonloc::ConcreteInt MinVal{MinValType};
  nonloc::ConcreteInt MaxVal{MaxValType};

  SValBuilder &SVB = C.getSValBuilder();
  ProgramStateRef State = C.getState();
  SVal IsLeMax = SVB.evalBinOp(State, BO_LE, RetVal, MaxVal, Res);
  SVal IsGeMin = SVB.evalBinOp(State, BO_GE, RetVal, MinVal, Res);

  auto [MayNotOverflow, MayOverflow] =
      State->assume(IsLeMax.castAs<DefinedOrUnknownSVal>());
  auto [MayNotUnderflow, MayUnderflow] =
      State->assume(IsGeMin.castAs<DefinedOrUnknownSVal>());

  return {MayOverflow || MayUnderflow, MayNotOverflow && MayNotUnderflow};
}

void BuiltinFunctionChecker::handleOverflowBuiltin(const CallEvent &Call,
                                                   CheckerContext &C,
                                                   BinaryOperator::Opcode Op,
                                                   QualType ResultType) const {
  // Calling a builtin with an incorrect argument count produces compiler error.
  assert(Call.getNumArgs() == 3);

  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();
  const Expr *CE = Call.getOriginExpr();

  SVal Arg1 = Call.getArgSVal(0);
  SVal Arg2 = Call.getArgSVal(1);

  SVal RetValMax = SVB.evalBinOp(State, Op, Arg1, Arg2,
                                 getSufficientTypeForOverflowOp(C, ResultType));
  SVal RetVal = SVB.evalBinOp(State, Op, Arg1, Arg2, ResultType);

  auto [Overflow, NotOverflow] = checkOverflow(C, RetValMax, ResultType);
  if (NotOverflow) {
    ProgramStateRef StateNoOverflow =
        State->BindExpr(CE, C.getLocationContext(), SVB.makeTruthVal(false));

    if (auto L = Call.getArgSVal(2).getAs<Loc>()) {
      StateNoOverflow =
          StateNoOverflow->bindLoc(*L, RetVal, C.getLocationContext());

      // Propagate taint if any of the argumets were tainted
      if (isTainted(State, Arg1) || isTainted(State, Arg2))
        StateNoOverflow = addTaint(StateNoOverflow, *L);
    }

    C.addTransition(
        StateNoOverflow,
        createBuiltinNoOverflowNoteTag(
            C, /*BothFeasible=*/NotOverflow && Overflow, Arg1, Arg2, RetVal));
  }

  if (Overflow) {
    C.addTransition(
        State->BindExpr(CE, C.getLocationContext(), SVB.makeTruthVal(true)),
        createBuiltinOverflowNoteTag(C));
  }
}

bool BuiltinFunctionChecker::isBuiltinLikeFunction(
    const CallEvent &Call) const {
  const auto *FD = llvm::dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD || FD->getNumParams() != 1)
    return false;

  if (QualType RetTy = FD->getReturnType();
      !RetTy->isPointerType() && !RetTy->isReferenceType())
    return false;

  if (QualType ParmTy = FD->getParamDecl(0)->getType();
      !ParmTy->isPointerType() && !ParmTy->isReferenceType())
    return false;

  return BuiltinLikeStdFunctions.contains(Call);
}

bool BuiltinFunctionChecker::evalCall(const CallEvent &Call,
                                      CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return false;

  const LocationContext *LCtx = C.getLocationContext();
  const Expr *CE = Call.getOriginExpr();

  if (isBuiltinLikeFunction(Call)) {
    C.addTransition(state->BindExpr(CE, LCtx, Call.getArgSVal(0)));
    return true;
  }

  unsigned BI = FD->getBuiltinID();

  switch (BI) {
  default:
    return false;
  case Builtin::BI__builtin_mul_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
    handleOverflowBuiltin(Call, C, BO_Mul,
                          getOverflowBuiltinResultType(Call, C, BI));
    return true;
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
    handleOverflowBuiltin(Call, C, BO_Sub,
                          getOverflowBuiltinResultType(Call, C, BI));
    return true;
  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
    handleOverflowBuiltin(Call, C, BO_Add,
                          getOverflowBuiltinResultType(Call, C, BI));
    return true;
  case Builtin::BI__builtin_assume:
  case Builtin::BI__assume: {
    assert (Call.getNumArgs() > 0);
    SVal Arg = Call.getArgSVal(0);
    if (Arg.isUndef())
      return true; // Return true to model purity.

    state = state->assume(Arg.castAs<DefinedOrUnknownSVal>(), true);
    // FIXME: do we want to warn here? Not right now. The most reports might
    // come from infeasible paths, thus being false positives.
    if (!state) {
      C.generateSink(C.getState(), C.getPredecessor());
      return true;
    }

    C.addTransition(state);
    return true;
  }

  case Builtin::BI__builtin_unpredictable:
  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability:
  case Builtin::BI__builtin_assume_aligned:
  case Builtin::BI__builtin_addressof:
  case Builtin::BI__builtin_function_start: {
    // For __builtin_unpredictable, __builtin_expect,
    // __builtin_expect_with_probability and __builtin_assume_aligned,
    // just return the value of the subexpression.
    // __builtin_addressof is going from a reference to a pointer, but those
    // are represented the same way in the analyzer.
    assert (Call.getNumArgs() > 0);
    SVal Arg = Call.getArgSVal(0);
    C.addTransition(state->BindExpr(CE, LCtx, Arg));
    return true;
  }

  case Builtin::BI__builtin_dynamic_object_size:
  case Builtin::BI__builtin_object_size:
  case Builtin::BI__builtin_constant_p: {
    // This must be resolvable at compile time, so we defer to the constant
    // evaluator for a value.
    SValBuilder &SVB = C.getSValBuilder();
    SVal V = UnknownVal();
    Expr::EvalResult EVResult;
    if (CE->EvaluateAsInt(EVResult, C.getASTContext(), Expr::SE_NoSideEffects)) {
      // Make sure the result has the correct type.
      llvm::APSInt Result = EVResult.Val.getInt();
      BasicValueFactory &BVF = SVB.getBasicValueFactory();
      BVF.getAPSIntType(CE->getType()).apply(Result);
      V = SVB.makeIntVal(Result);
    }

    if (FD->getBuiltinID() == Builtin::BI__builtin_constant_p) {
      // If we didn't manage to figure out if the value is constant or not,
      // it is safe to assume that it's not constant and unsafe to assume
      // that it's constant.
      if (V.isUnknown())
        V = SVB.makeIntVal(0, CE->getType());
    }

    C.addTransition(state->BindExpr(CE, LCtx, V));
    return true;
  }
  }
}

void ento::registerBuiltinFunctionChecker(CheckerManager &mgr) {
  mgr.registerChecker<BuiltinFunctionChecker>();
}

bool ento::shouldRegisterBuiltinFunctionChecker(const CheckerManager &mgr) {
  return true;
}
