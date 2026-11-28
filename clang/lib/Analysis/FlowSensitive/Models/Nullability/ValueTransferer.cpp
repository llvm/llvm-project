//===-- ValueTransferer.cpp - Value nullability transfer --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueTransferer.h"
#include "PointerNullabilityMatchers.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullability.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullabilityAnalysis.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include <cassert>

namespace clang::dataflow::nullability {

using ast_matchers::MatchFinder;

PointerValue *ensureRawPointerHasValue(const Expr *E, Environment &Env) {
  if (!isSupportedRawPointerType(E->getType()))
    return nullptr;

  if (E->isPRValue()) {
    if (auto *Val = Env.get<PointerValue>(*E))
      return Val;
    auto *Val = cast<PointerValue>(Env.createValue(E->getType()));
    Env.setValue(*E, *Val);
    return Val;
  }

  StorageLocation *Loc = Env.getStorageLocation(*E);
  if (Loc == nullptr) {
    Loc = &Env.createStorageLocation(*E);
    Env.setStorageLocation(*E, *Loc);
  }
  if (auto *Val = Env.get<PointerValue>(*Loc))
    return Val;
  auto *Val = cast<PointerValue>(Env.createValue(E->getType()));
  Env.setValue(*Loc, *Val);
  return Val;
}

static NullabilityKind getPointerTypeNullability(const Expr *E) {
  return E->getType()->getNullability().value_or(NullabilityKind::Unspecified);
}

void initPointerFromTypeNullability(
    PointerValue &PointerVal, const Expr *E,
    TransferState<PointerNullabilityLattice> &State) {
  initPointerNullState(PointerVal, State.Env.getDataflowAnalysisContext(),
                       getPointerTypeNullability(E));
}

/// If the pointer value stored at `PointerLoc` has any "top" nullability
/// properties, creates a new pointer value referencing the same location with
/// the "top" properties unpacked into fresh atoms. Returns:
/// -  The unpacked pointer value if unpacking took place.
/// -  The original pointer value if no unpacking took place.
/// -  Null if `PointerLoc` is not associated with a value.
/// This is analogous to the unpacking done on `TopBoolValue`s in the framework.
static PointerValue *unpackPointerValue(StorageLocation &PointerLoc,
                                        Environment &Env) {
  auto *PointerVal = Env.get<PointerValue>(PointerLoc);
  if (!PointerVal)
    return nullptr;

  PointerNullState NullState = getPointerNullState(*PointerVal);
  if (NullState.FromNullable && NullState.IsNull)
    return PointerVal;

  auto &A = Env.getDataflowAnalysisContext().arena();

  if (NullState.FromNullable == nullptr)
    NullState.FromNullable = &A.makeAtomRef(A.makeAtom());
  if (NullState.IsNull == nullptr)
    NullState.IsNull = &A.makeAtomRef(A.makeAtom());

  auto &NewPointerVal = Env.create<PointerValue>(PointerVal->getPointeeLoc());
  initPointerNullState(NewPointerVal, Env.getDataflowAnalysisContext(),
                       NullState);
  Env.setValue(PointerLoc, NewPointerVal);

  return &NewPointerVal;
}

static void
transferNullPointer(const Expr *NullPointer, const MatchFinder::MatchResult &,
                    TransferState<PointerNullabilityLattice> &State) {
  if (auto *PointerVal = ensureRawPointerHasValue(NullPointer, State.Env)) {
    initNullPointer(*PointerVal, State.Env.getDataflowAnalysisContext());
  }
}

static void
transferNotNullPointer(const Expr *NotNullPointer,
                       const MatchFinder::MatchResult &,
                       TransferState<PointerNullabilityLattice> &State) {
  if (auto *PointerVal = ensureRawPointerHasValue(NotNullPointer, State.Env)) {
    initPointerNullState(*PointerVal, State.Env.getDataflowAnalysisContext(),
                         NullabilityKind::NonNull);
  }
}

static void transferPointer(const Expr *PointerExpr,
                            const MatchFinder::MatchResult &Result,
                            TransferState<PointerNullabilityLattice> &State) {
  auto *PointerVal = ensureRawPointerHasValue(PointerExpr, State.Env);
  if (!PointerVal)
    return;

  // A pointer bitcast is null iff its operand is null, so seed the result
  // pointer's null-state from the operand *before* falling back to type-based
  // initialization below. This preserves the operand's flow-sensitive
  // null-state across the cast.
  //
  // We copy only the null-state and keep the result pointer's own pointee
  // storage location, since the operand and result pointee types can differ
  // across a bitcast (e.g. `void*` -> `T*`, or base -> derived).
  if (const auto *Cast = dyn_cast<CastExpr>(PointerExpr);
      Cast &&
      (Cast->getCastKind() == CK_BitCast ||
       Cast->getCastKind() == CK_LValueBitCast) &&
      isSupportedRawPointerType(Cast->getSubExpr()->getType()) &&
      isSupportedRawPointerType(Cast->getType()) &&
      !hasPointerNullState(*PointerVal)) {
    if (const PointerValue *SrcVal =
            getRawPointerValue(Cast->getSubExpr(), State.Env);
        SrcVal != nullptr && hasPointerNullState(*SrcVal)) {
      initPointerNullState(*PointerVal, State.Env.getDataflowAnalysisContext(),
                           getPointerNullState(*SrcVal));
    }
  }

  initPointerFromTypeNullability(*PointerVal, PointerExpr, State);

  if (const auto *Cast = dyn_cast<CastExpr>(PointerExpr);
      Cast && Cast->getCastKind() == CK_LValueToRValue) {
    // Forward the modeled `PointerValue` from the glvalue sub-expression's
    // storage location. The operand and result share the same pointee type
    // here, so forwarding the whole value is safe.
    if (StorageLocation *Loc =
            State.Env.getStorageLocation(*Cast->getSubExpr())) {
      if (PointerValue *Val = unpackPointerValue(*Loc, State.Env)) {
        State.Env.setValue(*PointerExpr, *Val);
      }
    }
  }
}

CFGMatchSwitch<TransferState<PointerNullabilityLattice>>
buildValueTransferer() {
  // The value transfer functions must establish:
  // - if we're transferring over an Expr
  // - and the Expr has a supported pointer type
  // - and the Expr's value is modeled by the framework (or this analysis)
  // - then the PointerValue has nullability properties (is_null/from_nullable)
  return CFGMatchSwitchBuilder<TransferState<PointerNullabilityLattice>>()
      // Handles initialization of the null states of pointers.
      .CaseOfCFGStmt<Expr>(isAddrOf(), transferNotNullPointer)
      .CaseOfCFGStmt<Expr>(isNullPointerLiteral(), transferNullPointer)
      .CaseOfCFGStmt<Expr>(isPointerExpr(), transferPointer)
      .Build();
}

} // namespace clang::dataflow::nullability
