//===- bolt/Passes/PettisAndHansen.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file implements Pettis and Hansen code-layout algorithm.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/HFSort.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <unordered_map>

#define DEBUG_TYPE "hfsort"

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;

namespace {
class ClusterArc {
public:
  ClusterArc(Cluster *Ca, Cluster *Cb, double W = 0)
      : C1(std::min(Ca, Cb)), C2(std::max(Ca, Cb)), Weight(W) {}

  friend bool operator==(const ClusterArc &Lhs, const ClusterArc &Rhs) {
    return Lhs.C1 == Rhs.C1 && Lhs.C2 == Rhs.C2;
  }

  Cluster *const C1;
  Cluster *const C2;
  mutable double Weight;
};

class ClusterArcHash {
public:
  int64_t operator()(const ClusterArc &Arc) const {
    std::hash<int64_t> Hasher;
    return hashCombine(Hasher(int64_t(Arc.C1)), int64_t(Arc.C2));
  }
};

using ClusterArcSet = std::unordered_set<ClusterArc, ClusterArcHash>;

void orderFuncs(const CallGraph &Cg, Cluster *C1, Cluster *C2) {
  NodeId C1head = C1->targets().front();
  NodeId C1tail = C1->targets().back();
  NodeId C2head = C2->targets().front();
  NodeId C2tail = C2->targets().back();

  double C1headC2head = 0;
  double C1headC2tail = 0;
  double C1tailC2head = 0;
  double C1tailC2tail = 0;

  for (const Arc &Arc : Cg.arcs()) {
    if ((Arc.src() == C1head && Arc.dst() == C2head) ||
        (Arc.dst() == C1head && Arc.src() == C2head))
      C1headC2head += Arc.weight();
    else if ((Arc.src() == C1head && Arc.dst() == C2tail) ||
             (Arc.dst() == C1head && Arc.src() == C2tail))
      C1headC2tail += Arc.weight();
    else if ((Arc.src() == C1tail && Arc.dst() == C2head) ||
             (Arc.dst() == C1tail && Arc.src() == C2head))
      C1tailC2head += Arc.weight();
    else if ((Arc.src() == C1tail && Arc.dst() == C2tail) ||
             (Arc.dst() == C1tail && Arc.src() == C2tail))
      C1tailC2tail += Arc.weight();
  }

  const double Max = std::max(std::max(C1headC2head, C1headC2tail),
                              std::max(C1tailC2head, C1tailC2tail));

  if (C1headC2head == Max) {
    // flip C1
    C1->reverseTargets();
  } else if (C1headC2tail == Max) {
    // flip C1 C2
    C1->reverseTargets();
    C2->reverseTargets();
  } else if (C1tailC2tail == Max) {
    // flip C2
    C2->reverseTargets();
  }
}
} // namespace

std::vector<Cluster> pettisAndHansen(const CallGraph &Cg) {
  // indexed by NodeId, keeps its current cluster
  std::vector<Cluster *> FuncCluster(Cg.numNodes(), nullptr);
  std::vector<Cluster> Clusters;
  std::vector<NodeId> Funcs;

  Clusters.reserve(Cg.numNodes());

  for (NodeId F = 0; F < Cg.numNodes(); F++) {
    if (Cg.samples(F) == 0)
      continue;
    Clusters.emplace_back(F, Cg.getNode(F));
    FuncCluster[F] = &Clusters.back();
    Funcs.push_back(F);
  }

  ClusterArcSet Carcs;

  auto insertOrInc = [&](Cluster *C1, Cluster *C2, double Weight) {
    auto Res = Carcs.emplace(C1, C2, Weight);
    if (!Res.second)
      Res.first->Weight += Weight;
  };

  // Create a std::vector of cluster arcs

  for (const Arc &Arc : Cg.arcs()) {
    if (Arc.weight() == 0)
      continue;

    Cluster *const S = FuncCluster[Arc.src()];
    Cluster *const D = FuncCluster[Arc.dst()];

    // ignore if s or d is nullptr

    if (S == nullptr || D == nullptr)
      continue;

    // ignore self-edges

    if (S == D)
      continue;

    insertOrInc(S, D, Arc.weight());
  }

  // Find an arc with max weight and merge its nodes

  while (!Carcs.empty()) {
    auto Maxpos =
        std::max_element(Carcs.begin(), Carcs.end(),
                         [&](const ClusterArc &Carc1, const ClusterArc &Carc2) {
                           return Carc1.Weight < Carc2.Weight;
                         });

    ClusterArc Max = *Maxpos;
    Carcs.erase(Maxpos);

    Cluster *const C1 = Max.C1;
    Cluster *const C2 = Max.C2;

    if (C1->size() + C2->size() > MaxClusterSize)
      continue;

    if (C1->frozen() || C2->frozen())
      continue;

    // order functions and merge cluster

    orderFuncs(Cg, C1, C2);

    LLVM_DEBUG(dbgs() << format("merging %s -> %s: %.1f\n",
                                C2->toString().c_str(), C1->toString().c_str(),
                                Max.Weight));

    // update carcs: merge C1arcs to C2arcs

    std::unordered_map<ClusterArc, Cluster *, ClusterArcHash> C2arcs;
    for (const ClusterArc &Carc : Carcs) {
      if (Carc.C1 == C2)
        C2arcs.emplace(Carc, Carc.C2);
      if (Carc.C2 == C2)
        C2arcs.emplace(Carc, Carc.C1);
    }

    for (auto It : C2arcs) {
      Cluster *const C = It.second;
      ClusterArc const C2arc = It.first;

      insertOrInc(C, C1, C2arc.Weight);
      Carcs.erase(C2arc);
    }

    // update FuncCluster

    for (NodeId F : C2->targets())
      FuncCluster[F] = C1;

    C1->merge(*C2, Max.Weight);
    C2->clear();
  }

  // Return the set of Clusters that are left, which are the ones that
  // didn't get merged.

  std::set<Cluster *> LiveClusters;
  std::vector<Cluster> OutClusters;

  for (NodeId Fid : Funcs)
    LiveClusters.insert(FuncCluster[Fid]);
  for (Cluster *C : LiveClusters)
    OutClusters.push_back(std::move(*C));

  llvm::sort(OutClusters, compareClustersDensity);

  return OutClusters;
}

} // namespace bolt
} // namespace llvm
