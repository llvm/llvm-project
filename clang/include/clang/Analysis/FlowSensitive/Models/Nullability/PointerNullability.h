//===-- PointerNullability.h - Pointer nullability properties ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file extends the dataflow framework's Value model to track nullability.
// We attach two boolean properties to each modeled pointer value:
//  - is_null: whether the pointer may actually be null
//    If this is false, dereferencing is safe.
//  - from_nullable: whether the originating expression was considered nullable
//    (e.g. a nullptr literal, or a reference to a Nullable-annotated variable)
//    If this is false, dereferencing may be safe: we don't know the contract.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITY_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITY_H

#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/Specifiers.h"
#include <optional>

namespace clang::dataflow::nullability {

/// Is this exactly a pointer type that we track outer nullability for?
/// This unwraps sugar, i.e. it looks at the canonical type.
///
/// (For now, only regular `PointerType`s. Support for smart pointers will be
/// added later; code that will need to handle them already uses this function
/// rather than `isSupportedRawPointerType()`, so it won't need to change. In
/// future we should consider supporting pointer-to-member, ObjC pointers, etc).
bool isSupportedPointerType(QualType T);

/// Is this exactly a raw (non-smart) pointer type that we track outer
/// nullability for?
/// This unwraps sugar, i.e. it looks at the canonical type.
bool isSupportedRawPointerType(QualType T);

/// Returns the `PointerValue` allocated to `PointerExpr` if available.
/// Otherwise, returns nullptr.
PointerValue *getRawPointerValue(const Expr *PointerExpr,
                                 const Environment &Env);

/// Returns the `PointerValue` for a raw pointer expression, if available.
/// Otherwise, returns nullptr.
///
/// This is intended to support both raw and smart pointer expressions.
/// Smart pointers are not supported yet, so for now this is the same as
/// `getRawPointerValue()`.
///
/// Use this function only if the expression can actually be either a raw or
/// smart pointer; otherwise, use `getRawPointerValue()`.
PointerValue *getPointerValue(const Expr *PointerExpr, const Environment &Env);

// Returns true if the pointer has all properties necessary for representing
// complete nullness information.
// Otherwise, returns false.
//
// Pointers that are the value of some expression always have null state once
// that expression has been analyzed. Other pointers, like the values of unused
// parameters, may lack this state. This state is only set by
// PointerNullabilityAnalysis, not by the dataflow framework.
bool hasPointerNullState(const PointerValue &PointerVal);

/// The properties representing nullness information for a pointer.
///
/// We attach these properties to every PointerValue taken by an expression.
///
/// A null pointer for `FromNullable` or `IsNull` represents "top", i.e. we have
/// no information on this property.
struct PointerNullState {
  /// Did the pointer come from a known-nullable source?
  const Formula *FromNullable;
  /// Is the pointer's value null?
  const Formula *IsNull;
  // These are independent: sources with unknown nullability can yield nullptr!
};

/// Returns the properties representing the nullness information of a pointer.
PointerNullState getPointerNullState(const PointerValue &PointerVal);

/// Creates the nullness properties on `PointerVal` if not already initialised.
///
/// We call this when the framework produces a PointerValue for an expression.
/// This ensures that the variable has usable "from nullable" and "is null"
/// boolean variables, and that they are constrained based on the *original*
/// source of the PointerValue.
///
/// For example:
///    int *_Null_unspecified x = getNullable();
///                               ~~~~~~~~~~~ <-- initPointerNullState(Nullable)
///    *x;
///     ~ <-- initPointerNullState(Unspecified) - no effect, already initialized
///
/// (We assume that the framework will not provide the same pointer from
/// different initial sources, so the `Source` nullability is the same
/// regardless of block evaluation order).
void initPointerNullState(PointerValue &PointerVal,
                          DataflowAnalysisContext &Ctx,
                          std::optional<NullabilityKind> Source = std::nullopt);

/// Initializes the nullness properties on `PointerVal` from `State`.
///
/// This overload may only be called on a freshly created `PointerValue` that
/// does not yet have nullability properties.
void initPointerNullState(PointerValue &PointerVal,
                          DataflowAnalysisContext &Ctx, PointerNullState State);

/// Variant of initPointerNullState, where the pointer is guaranteed null.
void initNullPointer(PointerValue &PointerVal, DataflowAnalysisContext &Ctx);

/// Returns true if `NullState` is known to be null or is from a nullable source
/// and may be null.
bool isNullable(PointerNullState NullState, const Environment &Env);

/// Returns true if `PointerVal` is known to be null or is from a nullable
/// source and may be null. Convenience overload for `isNullable` above.
bool isNullable(const PointerValue &PointerVal, const Environment &Env);

/// Returns the strongest provable assertion we can make for the given null
/// state. If a pointer with that state may not be null, returns Nonnull. If it
/// is known to be null or is from a nullable source and may be null, returns
/// Nullable. Otherwise, returns Unspecified.
NullabilityKind getNullability(PointerNullState PointerNullState,
                               const Environment &Env);

/// Returns the strongest provable assertion we can make about `PointerVal`.
NullabilityKind getNullability(const PointerValue &PointerVal,
                               const Environment &Env);

/// Returns the strongest provable assertion we can make about the value of
/// `E` in `Env`.
NullabilityKind getNullability(const Expr *E, const Environment &Env);

} // namespace clang::dataflow::nullability

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITY_H
