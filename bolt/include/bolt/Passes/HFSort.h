//===- bolt/Passes/HFSort.h - Cluster functions by hotness ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of HFSort algorithm for function ordering:
// https://research.fb.com/wp-content/uploads/2017/01/cgo2017-hfsort-final1.pdf
//
// Cluster functions by hotness.  There are three clustering algorithms:
// 1. clusterize
// 2. pettisAndHansen
// 3. randomClusters
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_HFSORT_H
#define BOLT_PASSES_HFSORT_H

#include "bolt/Core/CallGraph.h"

#include <string>
#include <vector>

namespace llvm {
namespace bolt {

class Cluster {
public:
  Cluster(CallGraph::NodeId Id, const CallGraph::Node &F);
  Cluster(const std::vector<CallGraph::NodeId> &Nodes, const CallGraph &Cg);

  std::string toString() const;
  double density() const { return Density; }
  uint64_t samples() const { return Samples; }
  uint32_t size() const { return Size; }
  bool frozen() const { return Frozen; }
  void freeze() { Frozen = true; }
  void merge(const Cluster &Other, const double Aw = 0);
  void merge(const Cluster &Other,
             const std::vector<CallGraph::NodeId> &Targets_);
  void clear();
  size_t numTargets() const { return Targets.size(); }
  const std::vector<CallGraph::NodeId> &targets() const { return Targets; }
  CallGraph::NodeId target(size_t N) const { return Targets[N]; }
  void reverseTargets();
  bool hasId() const { return Id != -1u; }
  void setId(uint32_t NewId) {
    assert(!hasId());
    Id = NewId;
  }
  uint32_t id() const {
    assert(hasId());
    return Id;
  }

private:
  uint32_t Id{-1u};
  std::vector<CallGraph::NodeId> Targets;
  uint64_t Samples{0};
  uint32_t Size{0};
  double Density{0.0};
  bool Frozen{false}; // not a candidate for merging
};

// Maximum size of a cluster, in bytes.
constexpr uint32_t MaxClusterSize = 1 << 20;

// Size of a huge page in bytes.
constexpr uint32_t HugePageSize = 2 << 20;

inline bool compareClustersDensity(const Cluster &C1, const Cluster &C2) {
  return C1.density() > C2.density();
}

/*
 * Cluster functions in order to minimize call distance.
 */
std::vector<Cluster> clusterize(const CallGraph &Cg);

/*
 * Pettis-Hansen code layout algorithm
 * reference: K. Pettis and R. C. Hansen, "Profile Guided Code Positioning",
 * PLDI '90
 */
std::vector<Cluster> pettisAndHansen(const CallGraph &Cg);

/* Group functions into clusters randomly. */
std::vector<Cluster> randomClusters(const CallGraph &Cg);

} // end namespace bolt
} // end namespace llvm

#endif // BOLT_PASSES_HFSORT_H
