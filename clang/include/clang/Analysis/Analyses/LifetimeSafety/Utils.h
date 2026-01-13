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

/// Computes the union of two ImmutableSets.
template <typename T>
static llvm::ImmutableSet<T> join(llvm::ImmutableSet<T> A,
                                  llvm::ImmutableSet<T> B,
                                  typename llvm::ImmutableSet<T>::Factory &F) {
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  for (const T &E : B)
    A = F.add(A, E);
  return A;
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

/// Computes the key-wise union of two ImmutableMaps.
// TODO(opt): This key-wise join is a performance bottleneck. A more
// efficient merge could be implemented using a Patricia Trie or HAMT
// instead of the current AVL-tree-based ImmutableMap.
template <typename K, typename V, typename Joiner>
static llvm::ImmutableMap<K, V>
join(const llvm::ImmutableMap<K, V> &A, const llvm::ImmutableMap<K, V> &B,
     typename llvm::ImmutableMap<K, V>::Factory &F, Joiner JoinValues,
     JoinKind Kind) {
  if (A.getHeight() < B.getHeight())
    return join(B, A, F, JoinValues, Kind);

  // For each element in B, join it with the corresponding element in A
  // (or with an empty value if it doesn't exist in A).
  llvm::ImmutableMap<K, V> Res = A;
  for (const auto &Entry : B) {
    const K &Key = Entry.first;
    const V &ValB = Entry.second;
    Res = F.add(Res, Key, JoinValues(A.lookup(Key), &ValB));
  }
  if (Kind == JoinKind::Symmetric) {
    for (const auto &Entry : A) {
      const K &Key = Entry.first;
      const V &ValA = Entry.second;
      if (!B.contains(Key))
        Res = F.add(Res, Key, JoinValues(&ValA, nullptr));
    }
  }
  return Res;
}
} // namespace clang::lifetimes::internal::utils

namespace llvm {
template <typename Tag>
struct DenseMapInfo<clang::lifetimes::internal::utils::ID<Tag>> {
  using ID = clang::lifetimes::internal::utils::ID<Tag>;

  static inline ID getEmptyKey() {
    return {DenseMapInfo<uint32_t>::getEmptyKey()};
  }

  static inline ID getTombstoneKey() {
    return {DenseMapInfo<uint32_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(const ID &Val) {
    return DenseMapInfo<uint32_t>::getHashValue(Val.Value);
  }

  static bool isEqual(const ID &LHS, const ID &RHS) { return LHS == RHS; }
};
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_UTILS_H
