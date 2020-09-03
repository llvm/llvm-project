//=== BuiltinFunctionChecker.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker evaluates clang builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicSize.h"

using namespace clang;
using namespace ento;

namespace {

class BuiltinFunctionChecker : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
};

}

bool BuiltinFunctionChecker::evalCall(const CallEvent &Call,
                                      CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return false;

  const LocationContext *LCtx = C.getLocationContext();
  const Expr *CE = Call.getOriginExpr();

  switch (FD->getBuiltinID()) {
  default:
    return false;

  case Builtin::BI__builtin_assume: {
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
  case Builtin::BI__builtin_addressof: {
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

  case Builtin::BI__builtin_alloca_with_align:
  case Builtin::BI__builtin_alloca: {
    // FIXME: Refactor into StoreManager itself?
    MemRegionManager& RM = C.getStoreManager().getRegionManager();
    const AllocaRegion* R =
      RM.getAllocaRegion(CE, C.blockCount(), C.getLocationContext());

    // Set the extent of the region in bytes. This enables us to use the
    // SVal of the argument directly. If we save the extent in bits, we
    // cannot represent values like symbol*8.
    auto Size = Call.getArgSVal(0);
    if (Size.isUndef())
      return true; // Return true to model purity.

    SValBuilder& svalBuilder = C.getSValBuilder();
    DefinedOrUnknownSVal DynSize = getDynamicSize(state, R, svalBuilder);
    DefinedOrUnknownSVal DynSizeMatchesSizeArg =
        svalBuilder.evalEQ(state, DynSize, Size.castAs<DefinedOrUnknownSVal>());
    state = state->assume(DynSizeMatchesSizeArg, true);
    assert(state && "The region should not have any previous constraints");

    C.addTransition(state->BindExpr(CE, LCtx, loc::MemRegionVal(R)));
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
