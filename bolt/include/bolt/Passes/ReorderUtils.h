//===- bolt/Passes/ReorderUtils.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains classes and functions to assist function and basic block
// reordering.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REORDER_UTILS_H
#define BOLT_PASSES_REORDER_UTILS_H

#include <vector>

#include "llvm/ADT/BitVector.h"

namespace llvm {
namespace bolt {

// This class maintains adjacency information for all Clusters being
// processed. It is used for visiting all neighbors of any given Cluster
// while merging pairs of Clusters. Every Cluster must implement the id() method
template <typename Cluster> class AdjacencyMatrix {
public:
  explicit AdjacencyMatrix(size_t Size) : Bits(Size, BitVector(Size, false)) {}

  void initialize(std::vector<Cluster *> &_Clusters) { Clusters = _Clusters; }

  template <typename F> void forAllAdjacent(const Cluster *C, F Func) const {
    const_cast<AdjacencyMatrix *>(this)->forallAdjacent(C, Func);
  }

  template <typename F> void forAllAdjacent(const Cluster *C, F Func) {
    for (int I : Bits[C->id()].set_bits())
      Func(Clusters[I]);
  }

  /// Merge adjacency info from cluster B into cluster A.  Info for cluster B is
  /// left in an undefined state.
  void merge(const Cluster *A, const Cluster *B) {
    Bits[A->id()] |= Bits[B->id()];
    Bits[A->id()][A->id()] = false;
    Bits[A->id()][B->id()] = false;
    Bits[B->id()][A->id()] = false;
    for (int I : Bits[B->id()].set_bits()) {
      Bits[I][A->id()] = true;
      Bits[I][B->id()] = false;
    }
  }

  void set(const Cluster *A, const Cluster *B) { set(A, B, true); }

private:
  void set(const Cluster *A, const Cluster *B, bool Value) {
    assert(A != B);
    Bits[A->id()][B->id()] = Value;
    Bits[B->id()][A->id()] = Value;
  }

  std::vector<Cluster *> Clusters;
  std::vector<BitVector> Bits;
};

// This class holds cached results of specified type for a pair of Clusters.
// It can invalidate all cache entries associated with a given Cluster.
template <typename Cluster, typename ValueType> class ClusterPairCache {
public:
  explicit ClusterPairCache(size_t Size)
      : Size(Size), Cache(Size * Size), Valid(Size * Size, false) {}

  bool contains(const Cluster *First, const Cluster *Second) const {
    return Valid[index(First, Second)];
  }

  ValueType get(const Cluster *First, const Cluster *Second) const {
    assert(contains(First, Second));
    return Cache[index(First, Second)];
  }

  void set(const Cluster *First, const Cluster *Second, ValueType Value) {
    const size_t Index = index(First, Second);
    Cache[Index] = Value;
    Valid[Index] = true;
  }

  void invalidate(const Cluster *C) {
    Valid.reset(C->id() * Size, (C->id() + 1) * Size);
    for (size_t Id = 0; Id < Size; Id++)
      Valid.reset((Id * Size) + C->id());
  }

private:
  size_t index(const Cluster *First, const Cluster *Second) const {
    return (First->id() * Size) + Second->id();
  }

  size_t Size;
  std::vector<ValueType> Cache;
  BitVector Valid;
};

// This class holds cached results of specified type for a pair of Clusters.
// It can invalidate all cache entries associated with a given Cluster.
// The functions set, get and contains are thread safe when called with
// distinct keys.
template <typename Cluster, typename ValueType>
class ClusterPairCacheThreadSafe {
public:
  explicit ClusterPairCacheThreadSafe(size_t Size)
      : Size(Size), Cache(Size * Size), Valid(Size * Size, false) {}

  bool contains(const Cluster *First, const Cluster *Second) const {
    return Valid[index(First, Second)];
  }

  ValueType get(const Cluster *First, const Cluster *Second) const {
    assert(contains(First, Second));
    return Cache[index(First, Second)];
  }

  void set(const Cluster *First, const Cluster *Second, ValueType Value) {
    const size_t Index = index(First, Second);
    Cache[Index] = Value;
    Valid[Index] = true;
  }

  void invalidate(const Cluster *C) {
    for (size_t Idx = C->id() * Size; Idx < (C->id() + 1) * Size; Idx++)
      Valid[Idx] = false;

    for (size_t Id = 0; Id < Size; Id++)
      Valid[(Id * Size) + C->id()] = false;
  }

private:
  size_t Size;
  std::vector<ValueType> Cache;
  BitVector Valid;

  size_t index(const Cluster *First, const Cluster *Second) const {
    return (First->id() * Size) + Second->id();
  }
};

} // namespace bolt
} // namespace llvm

#endif
