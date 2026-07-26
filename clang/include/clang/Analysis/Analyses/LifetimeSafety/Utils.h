//===- Utils.h - Utility Functions for Lifetime Safety --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file provides utilities for the lifetime safety analysis, including
// join operations for LLVM's immutable data structures.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_UTILS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_UTILS_H

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"

namespace clang::lifetimes::internal::utils {

/// A generic, type-safe wrapper for an ID, distinguished by its `Tag` type.
/// Used for giving ID to loans and origins.
template <typename Tag> struct ID {
  uint32_t Value = 0;

  bool operator==(const ID<Tag> &Other) const { return Value == Other.Value; }
  bool operator!=(const ID<Tag> &Other) const { return !(*this == Other); }
  bool operator<(const ID<Tag> &Other) const { return Value < Other.Value; }
  ID<Tag> operator++(int) {
    ID<Tag> Tmp = *this;
    ++Value;
    return Tmp;
  }
  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddInteger(Value);
  }
};

/// The lifetime analyses do not benefit from canonicalizing their immutable
/// collections, so they opt out of it via these aliases.
template <typename T>
using SetTy =
    llvm::ImmutableSet<T, llvm::ImutContainerInfo<T>, /*Canonicalize=*/false>;

template <typename KeyT, typename ValT>
using MapTy = llvm::ImmutableMap<KeyT, ValT, llvm::ImutKeyValueInfo<KeyT, ValT>,
                                 /*Canonicalize=*/false>;

/// Computes the union of two ImmutableSets.
template <typename T>
SetTy<T> join(const SetTy<T> &A_ref, const SetTy<T> &B_ref, typename SetTy<T>::Factory &F) {
  if (A_ref.getRootWithoutRetain() == B_ref.getRootWithoutRetain())
    return A_ref;
  SetTy<T> A = A_ref;
  SetTy<T> B = B_ref;
  SetTy<T> Result = F.unionSets(A, B);
  if (Result.getRootWithoutRetain() == A_ref.getRootWithoutRetain())
    return A_ref;
  if (Result.getRootWithoutRetain() == B_ref.getRootWithoutRetain())
    return B_ref;
  return Result;
}

/// Describes the strategy for joining two `ImmutableMap` instances, primarily
/// differing in how they handle keys that are unique to one of the maps.
///
/// A `Symmetric` join is universally correct, while an `Asymmetric` join
/// serves as a performance optimization. The latter is applicable only when the
/// join operation possesses a left identity element, allowing for a more
/// efficient, one-sided merge.
enum class JoinKind {
  /// A symmetric join applies the `JoinValues` operation to keys unique to
  /// either map, ensuring that values from both maps contribute to the result.
  Symmetric,
  /// An asymmetric join preserves keys unique to the first map as-is, while
  /// applying the `JoinValues` operation only to keys unique to the second map.
  Asymmetric,
};

/// Computes the key-wise union of two ImmutableMaps in a single traversal
/// (see ImmutableMap::Factory::mergeWith), sharing subtrees the two maps do
/// not overlap. This assumes -- as the swap below already does -- that
/// JoinValues is commutative with a left identity, which holds for the
/// lifetime lattices.
template <typename KeyT, typename ValT, typename Joiner>
MapTy<KeyT, ValT> join(const MapTy<KeyT, ValT> &A_ref, const MapTy<KeyT, ValT> &B_ref,
                       typename MapTy<KeyT, ValT>::Factory &F,
                       Joiner JoinValues, JoinKind Kind) {
  if (A_ref.getRootWithoutRetain() == B_ref.getRootWithoutRetain())
    return A_ref;

  MapTy<KeyT, ValT> A = A_ref;
  MapTy<KeyT, ValT> B = B_ref;

  bool Swapped = false;
  if (A.getHeight() < B.getHeight()) {
    std::swap(A, B);
    Swapped = true;
  }

  using ValueTy = typename MapTy<KeyT, ValT>::value_type;
  auto Combine = [&JoinValues, Swapped](const ValueTy *AElem,
                               const ValueTy *BElem) -> std::pair<KeyT, ValT> {
    const ValueTy *OrigA = Swapped ? BElem : AElem;
    const ValueTy *OrigB = Swapped ? AElem : BElem;
    const KeyT &Key = AElem ? AElem->first : BElem->first;
    return std::pair<KeyT, ValT>(Key,
                                 JoinValues(OrigA ? &OrigA->second : nullptr,
                                            OrigB ? &OrigB->second : nullptr));
  };
  // Asymmetric keeps keys unique to either map as-is (valid because JoinValues
  // has a left identity); symmetric passes unmatched keys through JoinValues.
  // The lifetime joins are idempotent lattice joins, so pointer-identical
  // subtrees (common once one state is derived from the other) can be shared.
  MapTy<KeyT, ValT> Result = F.mergeWith(A, B, Combine,
                                         /*KeepUnmatched=*/Kind == JoinKind::Asymmetric,
                                         /*SkipShared=*/true);
  if (Result.getRootWithoutRetain() == A_ref.getRootWithoutRetain())
    return A_ref;
  if (Result.getRootWithoutRetain() == B_ref.getRootWithoutRetain())
    return B_ref;
  return Result;
}
} // namespace clang::lifetimes::internal::utils

namespace llvm {
template <typename Tag>
struct DenseMapInfo<clang::lifetimes::internal::utils::ID<Tag>> {
  using ID = clang::lifetimes::internal::utils::ID<Tag>;

  static unsigned getHashValue(const ID &Val) {
    return DenseMapInfo<uint32_t>::getHashValue(Val.Value);
  }

  static bool isEqual(const ID &LHS, const ID &RHS) { return LHS == RHS; }
};
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_UTILS_H
