//===-- SmartPointerAccessorCaching.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to help cache accessors for smart pointer
// like objects.
//
// These should be combined with CachedConstAccessorsLattice.
// Beyond basic const accessors, smart pointers may have the following two
// additional issues:
//
// 1) There may be multiple accessors for the same underlying object, e.g.
//    `operator->`, `operator*`, and `get`. Users may use a mixture of these
//    accessors, so the cache should unify them.
//
// 2) There may be non-const overloads of accessors. They are still safe to
//    cache, as they don't modify the container object.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SMARTPOINTERACCESSORCACHING_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SMARTPOINTERACCESSORCACHING_H

#include <cassert>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang::dataflow {

/// Matchers:
/// For now, these match on any class with an `operator*` or `operator->`
/// where the return types have a similar shape as std::unique_ptr
/// and std::optional.
///
/// - `*` returns a reference to a type `T`
/// - `->` returns a pointer to `T`
/// - `get` returns a pointer to `T`
/// - `value` returns a reference `T`
///
/// (1) The `T` should all match across the accessors (ignoring qualifiers).
///
/// (2) The specific accessor used in a call isn't required to be const,
///     but the class must have a const overload of each accessor.
///
/// For now, we don't have customization to ignore certain classes.
/// For example, if writing a ClangTidy check for `std::optional`, these
/// would also match `std::optional`. In order to have special handling
/// for `std::optional`, we assume the (Matcher, TransferFunction) case
/// with custom handling is ordered early so that these generic cases
/// do not trigger.
ast_matchers::StatementMatcher isSmartPointerLikeOperatorStar();
ast_matchers::StatementMatcher isSmartPointerLikeOperatorArrow();
ast_matchers::StatementMatcher isSmartPointerLikeValueMethodCall();
ast_matchers::StatementMatcher isSmartPointerLikeGetMethodCall();

// Common transfer functions.

/// Returns the "canonical" callee for smart pointer operators (`*` and `->`)
/// as a key for caching.
///
/// We choose `*` as the canonical one, since it needs a
/// StorageLocation anyway.
///
/// Note: there may be multiple `operator*` (one const, one non-const).
/// We pick the const one, which the above provided matchers require to exist.
const FunctionDecl *
getCanonicalSmartPointerLikeOperatorCallee(const CallExpr *CE);

/// A transfer function for `operator*` (and `value`) calls that can be
/// cached. Runs the `InitializeLoc` callback to initialize any new
/// StorageLocations.
///
/// Requirements:
///
/// - LatticeT should use the `CachedConstAccessorsLattice` mixin.
template <typename LatticeT>
void transferSmartPointerLikeCachedDeref(
    const CallExpr *DerefExpr, RecordStorageLocation *SmartPointerLoc,
    TransferState<LatticeT> &State,
    llvm::function_ref<void(StorageLocation &)> InitializeLoc);

/// A transfer function for `operator->` (and `get`) calls that can be cached.
/// Runs the `InitializeLoc` callback to initialize any new StorageLocations.
///
/// Requirements:
///
/// - LatticeT should use the `CachedConstAccessorsLattice` mixin.
template <typename LatticeT>
void transferSmartPointerLikeCachedGet(
    const CallExpr *GetExpr, RecordStorageLocation *SmartPointerLoc,
    TransferState<LatticeT> &State,
    llvm::function_ref<void(StorageLocation &)> InitializeLoc);

template <typename LatticeT>
void transferSmartPointerLikeCachedDeref(
    const CallExpr *DerefExpr, RecordStorageLocation *SmartPointerLoc,
    TransferState<LatticeT> &State,
    llvm::function_ref<void(StorageLocation &)> InitializeLoc) {
  if (State.Env.getStorageLocation(*DerefExpr) != nullptr)
    return;
  if (SmartPointerLoc == nullptr)
    return;

  const FunctionDecl *Callee = DerefExpr->getDirectCallee();
  if (Callee == nullptr)
    return;
  const FunctionDecl *CanonicalCallee =
      getCanonicalSmartPointerLikeOperatorCallee(DerefExpr);
  // This shouldn't happen, as we should at least find `Callee` itself.
  assert(CanonicalCallee != nullptr);
  if (CanonicalCallee != Callee) {
    // When using the provided matchers, we should always get a reference to
    // the same type.
    assert(CanonicalCallee->getReturnType()->isReferenceType() &&
           Callee->getReturnType()->isReferenceType());
    assert(CanonicalCallee->getReturnType()
               .getNonReferenceType()
               ->getCanonicalTypeUnqualified() ==
           Callee->getReturnType()
               .getNonReferenceType()
               ->getCanonicalTypeUnqualified());
  }

  StorageLocation &LocForValue =
      State.Lattice.getOrCreateConstMethodReturnStorageLocation(
          *SmartPointerLoc, CanonicalCallee, State.Env, InitializeLoc);
  State.Env.setStorageLocation(*DerefExpr, LocForValue);
}

template <typename LatticeT>
void transferSmartPointerLikeCachedGet(
    const CallExpr *GetExpr, RecordStorageLocation *SmartPointerLoc,
    TransferState<LatticeT> &State,
    llvm::function_ref<void(StorageLocation &)> InitializeLoc) {
  if (SmartPointerLoc == nullptr)
    return;

  const FunctionDecl *CanonicalCallee =
      getCanonicalSmartPointerLikeOperatorCallee(GetExpr);

  if (CanonicalCallee != nullptr) {
    auto &LocForValue =
        State.Lattice.getOrCreateConstMethodReturnStorageLocation(
            *SmartPointerLoc, CanonicalCallee, State.Env, InitializeLoc);
    State.Env.setValue(*GetExpr,
                       State.Env.template create<PointerValue>(LocForValue));
  } else {
    // Otherwise, just cache the pointer value as if it was a const accessor.
    Value *Val = State.Lattice.getOrCreateConstMethodReturnValue(
        *SmartPointerLoc, GetExpr, State.Env);
    if (Val == nullptr)
      return;
    State.Env.setValue(*GetExpr, *Val);
  }
}

} // namespace clang::dataflow

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SMARTPOINTERACCESSORCACHING_H
