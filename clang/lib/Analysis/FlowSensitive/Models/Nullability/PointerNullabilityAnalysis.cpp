//===-- PointerNullabilityAnalysis.cpp - Nullability analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullabilityAnalysis.h"
#include "ValueTransferer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullability.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LLVM.h"
#include <cassert>
#include <optional>

namespace clang::dataflow::nullability {

// If `Elt` is an expression of raw pointer type, ensures that it has a
// `PointerValue` associated with it. Also ensure that it has nullability
// state.
static void ensureRawPointerHasValueAndNullability(
    const CFGElement &Elt, Environment &Env,
    TransferState<PointerNullabilityLattice> &State) {
  std::optional<CFGStmt> S = Elt.getAs<CFGStmt>();
  if (!S)
    return;

  const Expr *E = dyn_cast<Expr>(S->getStmt());
  if (!E)
    return;

  if (PointerValue *PointerVal = ensureRawPointerHasValue(E, Env)) {
    if (!hasPointerNullState(*PointerVal)) {
      initPointerFromTypeNullability(*PointerVal, E, State);
    }
  }
}

PointerNullabilityAnalysis::PointerNullabilityAnalysis(ASTContext &Context,
                                                       Environment &Env)
    : DataflowAnalysis<PointerNullabilityAnalysis, PointerNullabilityLattice>(
          Context),
      ValueTransferer(buildValueTransferer()) {}

void PointerNullabilityAnalysis::transfer(const CFGElement &Elt,
                                          PointerNullabilityLattice &Lattice,
                                          Environment &Env) {
  TransferState<PointerNullabilityLattice> State(Lattice, Env);

  ValueTransferer(Elt, getASTContext(), State);
  ensureRawPointerHasValueAndNullability(Elt, Env, State);
}

static const Formula *mergeFormulas(const Formula *Bool1,
                                    const Environment &Env1,
                                    const Formula *Bool2,
                                    const Environment &Env2,
                                    Environment &MergedEnv) {
  if (Bool1 == Bool2)
    return Bool1;

  if (Bool1 == nullptr && Bool2 == nullptr)
    return nullptr;

  Arena &A = MergedEnv.arena();
  // If one side is Top (unconstrained), replace it with a fresh atom, which is
  // still unconstrained. Analogous to unpacking TopBoolValues in the framework.
  if (Bool1 == nullptr)
    Bool1 = &A.makeAtomRef(A.makeAtom());
  if (Bool2 == nullptr)
    Bool2 = &A.makeAtomRef(A.makeAtom());

  // If `Bool1` and `Bool2` are constrained to the same true / false value,
  // return that literal directly rather than delegating to `joinBoolValues`.
  // Otherwise, in the SAT bool model, the framework will allocate a fresh atom
  // and flow-condition constraint).
  if (Env1.proves(*Bool1)) {
    if (Env2.proves(*Bool2))
      return &A.makeLiteral(true);
  } else if (Env1.proves(A.makeNot(*Bool1)) && Env2.proves(A.makeNot(*Bool2))) {
    return &A.makeLiteral(false);
  }

  // Otherwise, let the boolean model decide how to join the two values (e.g.,
  // the SAT model uses information about which path was taken to associate the
  // result with `Bool1` and `Bool2`).
  return &MergedEnv
              .joinBoolValues(A.makeBoolValue(*Bool1), Env1,
                              A.makeBoolValue(*Bool2), Env2)
              .formula();
}

void PointerNullabilityAnalysis::join(QualType Type, const Value &Val1,
                                      const Environment &Env1,
                                      const Value &Val2,
                                      const Environment &Env2, Value &MergedVal,
                                      Environment &MergedEnv) {
  if (!isSupportedRawPointerType(Type))
    return;

  if (!hasPointerNullState(cast<PointerValue>(Val1)) ||
      !hasPointerNullState(cast<PointerValue>(Val2))) {
    // It can happen that we merge pointers without null state, if either or
    // both of the pointers has not appeared in an expression (and has not
    // otherwise been initialized with nullability properties) before the merge.
    // When the merged value appears in an expression, `transferPointer` will
    // take care of initializing it with nullability properties.
    return;
  }

  PointerNullState Nullability1 = getPointerNullState(cast<PointerValue>(Val1));
  PointerNullState Nullability2 = getPointerNullState(cast<PointerValue>(Val2));

  const Formula *FromNullable =
      mergeFormulas(Nullability1.FromNullable, Env1, Nullability2.FromNullable,
                    Env2, MergedEnv);
  const Formula *Null = mergeFormulas(Nullability1.IsNull, Env1,
                                      Nullability2.IsNull, Env2, MergedEnv);

  initPointerNullState(cast<PointerValue>(MergedVal),
                       MergedEnv.getDataflowAnalysisContext(),
                       {FromNullable, Null});
}

} // namespace clang::dataflow::nullability
