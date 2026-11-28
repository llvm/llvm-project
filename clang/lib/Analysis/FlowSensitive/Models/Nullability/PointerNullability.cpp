//===-- PointerNullability.cpp - Pointer nullability properties -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullability.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <optional>

namespace clang::dataflow::nullability {

/// The nullness information of a pointer is represented by two properties
/// which indicate if its source was nullable, and if its value is null.
constexpr llvm::StringLiteral kFromNullable = "from_nullable";
constexpr llvm::StringLiteral kNull = "is_null";

bool isSupportedPointerType(QualType T) { return isSupportedRawPointerType(T); }

bool isSupportedRawPointerType(QualType T) { return T->isPointerType(); }

PointerValue *getRawPointerValue(const Expr *PointerExpr,
                                 const Environment &Env) {
  return Env.get<PointerValue>(*PointerExpr);
}

PointerValue *getPointerValue(const Expr *PointerExpr, const Environment &Env) {
  QualType Ty = PointerExpr->getType();
  if (Ty->isNullPtrType() || isSupportedRawPointerType(Ty))
    return getRawPointerValue(PointerExpr, Env);
  return nullptr;
}

bool hasPointerNullState(const PointerValue &PointerVal) {
  return PointerVal.getProperty(kFromNullable) != nullptr &&
         PointerVal.getProperty(kNull) != nullptr;
}

PointerNullState getPointerNullState(const PointerValue &PointerVal) {
  Value *FromNullableProp = PointerVal.getProperty(kFromNullable);
  Value *NullProp = PointerVal.getProperty(kNull);

  assert(FromNullableProp != nullptr && NullProp != nullptr &&
         "PointerVal is missing null state!");

  return {
      isa<TopBoolValue>(FromNullableProp)
          ? nullptr
          : &cast<BoolValue>(FromNullableProp)->formula(),
      isa<TopBoolValue>(NullProp) ? nullptr
                                  : &cast<BoolValue>(NullProp)->formula(),
  };
}

static bool tryCreatePointerNullState(PointerValue &PointerVal, Arena &A,
                                      const Formula *FromNullable = nullptr,
                                      const Formula *IsNull = nullptr) {
  if (hasPointerNullState(PointerVal))
    return false;
  if (!FromNullable)
    FromNullable = &A.makeAtomRef(A.makeAtom());
  if (!IsNull)
    IsNull = &A.makeAtomRef(A.makeAtom());
  PointerVal.setProperty(kFromNullable, A.makeBoolValue(*FromNullable));
  PointerVal.setProperty(kNull, A.makeBoolValue(*IsNull));
  return true;
}

void initPointerNullState(PointerValue &PointerVal,
                          DataflowAnalysisContext &Ctx,
                          std::optional<NullabilityKind> Source) {
  auto &A = Ctx.arena();
  tryCreatePointerNullState(
      PointerVal, A,
      Source ? &A.makeLiteral(*Source == NullabilityKind::Nullable) : nullptr,
      Source == NullabilityKind::NonNull ? &A.makeLiteral(false) : nullptr);
}

void initPointerNullState(PointerValue &PointerVal,
                          DataflowAnalysisContext &Ctx,
                          PointerNullState State) {
  assert(!hasPointerNullState(PointerVal));

  auto &A = Ctx.arena();

  // Internally, we encode "top" as `TopBoolValue`.
  BoolValue &FromNullable = State.FromNullable != nullptr
                                ? A.makeBoolValue(*State.FromNullable)
                                : A.makeTopValue();
  BoolValue &IsNull = State.IsNull != nullptr ? A.makeBoolValue(*State.IsNull)
                                              : A.makeTopValue();
  PointerVal.setProperty(kFromNullable, FromNullable);
  PointerVal.setProperty(kNull, IsNull);
}

void initNullPointer(PointerValue &PointerVal, DataflowAnalysisContext &Ctx) {
  tryCreatePointerNullState(PointerVal, Ctx.arena(),
                            /*FromNullable=*/&Ctx.arena().makeLiteral(true),
                            /*IsNull=*/&Ctx.arena().makeLiteral(true));
}

bool isNullable(PointerNullState PointerNullState, const Environment &Env) {
  auto &A = Env.getDataflowAnalysisContext().arena();
  auto [FromNullable, Null] = PointerNullState;

  // A value is nullable if either of two things is true:
  // - The value is provably null under satisfiable flow conditions
  //   (a value from an unknown source becomes nullable if provably null)
  // - We got it from a nullable source and it may be actually null
  //   (if a value from a nullable source was checked, it's not nullable)
  //
  // Notably, a value from an unknown source that may be null, but is not
  // provably null, is not considered nullable.
  // Values from non-null sources are never considered nullable because being
  // able to prove them null can only occur under unsatisfiable flow conditions.
  if (Null) {
    if (Null->isLiteral(false))
      return false;

    if (Env.proves(*Null)) {
      // If we're in an environment with false flow conditions, we can prove
      // anything, but don't want to consider this value Nullable and end up
      // producing diagnostics in unreachable code.
      if (Env.proves(A.makeLiteral(false)))
        return false;
      return true;
    }
  }

  const Formula *NullableAndMaybeNull = &A.makeLiteral(true);
  if (FromNullable)
    NullableAndMaybeNull = &A.makeAnd(*NullableAndMaybeNull, *FromNullable);
  if (Null)
    NullableAndMaybeNull = &A.makeAnd(*NullableAndMaybeNull, *Null);

  return Env.allows(*NullableAndMaybeNull);
}

bool isNullable(const PointerValue &PointerVal, const Environment &Env) {
  return isNullable(getPointerNullState(PointerVal), Env);
}

NullabilityKind getNullability(PointerNullState PointerNullState,
                               const Environment &Env) {
  auto &A = Env.getDataflowAnalysisContext().arena();
  if (auto *Null = PointerNullState.IsNull) {
    if (Env.proves(A.makeNot(*Null)))
      return NullabilityKind::NonNull;
  }
  return isNullable(PointerNullState, Env) ? NullabilityKind::Nullable
                                           : NullabilityKind::Unspecified;
}

NullabilityKind getNullability(const PointerValue &PointerVal,
                               const Environment &Env) {
  return getNullability(getPointerNullState(PointerVal), Env);
}

NullabilityKind getNullability(const Expr *E, const Environment &Env) {
  if (PointerValue *P = getPointerValue(E, Env))
    return getNullability(*P, Env);
  return NullabilityKind::Unspecified;
}

} // namespace clang::dataflow::nullability
