//===- bolt/Passes/ReorderAlgorithm.h - Basic block reorderng ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to different basic block reordering algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REORDER_ALGORITHM_H
#define BOLT_PASSES_REORDER_ALGORITHM_H

#include "bolt/Core/BinaryFunction.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace llvm {

class raw_ostream;

namespace bolt {

/// Objects of this class implement various basic block clustering algorithms.
/// Basic block clusters are chains of basic blocks that should be laid out
/// in this order to maximize performace. These algorithms group basic blocks
/// into clusters using execution profile data and various heuristics.
class ClusterAlgorithm {
public:
  using ClusterTy = std::vector<BinaryBasicBlock *>;
  std::vector<ClusterTy> Clusters;
  std::vector<std::unordered_map<uint32_t, uint64_t>> ClusterEdges;
  std::vector<double> AvgFreq;

  /// Group the basic blocks in the given function into clusters stored in the
  /// Clusters vector. Also encode relative weights between two clusters in
  /// the ClusterEdges vector if requested. This vector is indexed by
  /// the clusters indices in the Clusters vector.
  virtual void clusterBasicBlocks(BinaryFunction &BF,
                                  bool ComputeEdges = false) = 0;

  /// Compute for each cluster its average execution frequency, that is
  /// the sum of average frequencies of its blocks (execution count / # instrs).
  /// The average frequencies are stored in the AvgFreq vector, index by the
  /// cluster indices in the Clusters vector.
  void computeClusterAverageFrequency(const BinaryContext &BC);

  /// Clear clusters and related info.
  virtual void reset();

  void printClusters() const;

  virtual ~ClusterAlgorithm() {}
};

/// Base class for a greedy clustering algorithm that selects edges in order
/// based on some heuristic and uses them to join basic blocks into clusters.
class GreedyClusterAlgorithm : public ClusterAlgorithm {
protected:
  // Represents an edge between two basic blocks, with source, destination, and
  // profile count.
  struct EdgeTy {
    const BinaryBasicBlock *Src;
    const BinaryBasicBlock *Dst;
    uint64_t Count;

    EdgeTy(const BinaryBasicBlock *Src, const BinaryBasicBlock *Dst,
           uint64_t Count)
        : Src(Src), Dst(Dst), Count(Count) {}

    void print(raw_ostream &OS) const;
  };

  struct EdgeHash {
    size_t operator()(const EdgeTy &E) const;
  };

  struct EdgeEqual {
    bool operator()(const EdgeTy &A, const EdgeTy &B) const;
  };

  // Virtual methods that allow custom specialization of the heuristic used by
  // the algorithm to select edges.
  virtual void initQueue(std::vector<EdgeTy> &Queue,
                         const BinaryFunction &BF) = 0;
  virtual void adjustQueue(std::vector<EdgeTy> &Queue,
                           const BinaryFunction &BF) = 0;
  virtual bool areClustersCompatible(const ClusterTy &Front,
                                     const ClusterTy &Back,
                                     const EdgeTy &E) const = 0;

  // Map from basic block to owning cluster index.
  using BBToClusterMapTy =
      std::unordered_map<const BinaryBasicBlock *, unsigned>;
  BBToClusterMapTy BBToClusterMap;

public:
  void clusterBasicBlocks(BinaryFunction &BF,
                          bool ComputeEdges = false) override;
  void reset() override;
};

/// This clustering algorithm is based on a greedy heuristic suggested by
/// Pettis and Hansen (PLDI '90).
class PHGreedyClusterAlgorithm : public GreedyClusterAlgorithm {
protected:
  void initQueue(std::vector<EdgeTy> &Queue, const BinaryFunction &BF) override;
  void adjustQueue(std::vector<EdgeTy> &Queue,
                   const BinaryFunction &BF) override;
  bool areClustersCompatible(const ClusterTy &Front, const ClusterTy &Back,
                             const EdgeTy &E) const override;
};

/// This clustering algorithm is based on a greedy heuristic that is a
/// modification of the heuristic suggested by Pettis (PLDI '90). It is
/// geared towards minimizing branches.
class MinBranchGreedyClusterAlgorithm : public GreedyClusterAlgorithm {
private:
  // Map from an edge to its weight which is used by the algorithm to sort the
  // edges.
  std::unordered_map<EdgeTy, int64_t, EdgeHash, EdgeEqual> Weight;

  // The weight of an edge is calculated as the win in branches if we choose
  // to layout this edge as a fall-through. For example, consider the edges
  //  A -> B with execution count 500,
  //  A -> C with execution count 100, and
  //  D -> B with execution count 150
  // where B, C are the only successors of A and A, D are the only predecessors
  // of B. Then if we choose to layout edge A -> B as a fallthrough, the win in
  // branches would be 500 - 100 - 150 = 250. That is the weight of edge A->B.
  int64_t calculateWeight(const EdgeTy &E, const BinaryFunction &BF) const;

protected:
  void initQueue(std::vector<EdgeTy> &Queue, const BinaryFunction &BF) override;
  void adjustQueue(std::vector<EdgeTy> &Queue,
                   const BinaryFunction &BF) override;
  bool areClustersCompatible(const ClusterTy &Front, const ClusterTy &Back,
                             const EdgeTy &E) const override;

public:
  void reset() override;
};

/// Objects of this class implement various basic block reordering algorithms.
/// Most of these algorithms depend on a clustering algorithm.
/// Here we have 3 conflicting goals as to how to layout clusters. If we want
/// to minimize jump offsets, we should put clusters with heavy inter-cluster
/// dependence as close as possible. If we want to maximize the probability
/// that all inter-cluster edges are predicted as not-taken, we should enforce
/// a topological order to make targets appear after sources, creating forward
/// branches. If we want to separate hot from cold blocks to maximize the
/// probability that unfrequently executed code doesn't pollute the cache, we
/// should put clusters in descending order of hotness.
class ReorderAlgorithm {
protected:
  std::unique_ptr<ClusterAlgorithm> CAlgo;

public:
  ReorderAlgorithm() {}
  explicit ReorderAlgorithm(std::unique_ptr<ClusterAlgorithm> CAlgo)
      : CAlgo(std::move(CAlgo)) {}

  using BasicBlockOrder = BinaryFunction::BasicBlockOrderType;

  /// Reorder the basic blocks of the given function and store the new order in
  /// the new Clusters vector.
  virtual void reorderBasicBlocks(BinaryFunction &BF,
                                  BasicBlockOrder &Order) const = 0;

  void setClusterAlgorithm(ClusterAlgorithm *CAlgo) {
    this->CAlgo.reset(CAlgo);
  }

  virtual ~ReorderAlgorithm() {}
};

/// Dynamic programming implementation for the TSP, applied to BB layout. Find
/// the optimal way to maximize weight during a path traversing all BBs. In
/// this way, we will convert the hottest branches into fall-throughs.
///
/// Uses exponential amount of memory on the number of basic blocks and should
/// only be used for small functions.
class TSPReorderAlgorithm : public ReorderAlgorithm {
public:
  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

/// Simple algorithm that groups basic blocks into clusters and then
/// lays them out cluster after cluster.
class OptimizeReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit OptimizeReorderAlgorithm(std::unique_ptr<ClusterAlgorithm> CAlgo)
      : ReorderAlgorithm(std::move(CAlgo)) {}

  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

/// This reorder algorithm tries to ensure that all inter-cluster edges are
/// predicted as not-taken, by enforcing a topological order to make
/// targets appear after sources, creating forward branches.
class OptimizeBranchReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit OptimizeBranchReorderAlgorithm(
      std::unique_ptr<ClusterAlgorithm> CAlgo)
      : ReorderAlgorithm(std::move(CAlgo)) {}

  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

/// This reorder tries to separate hot from cold blocks to maximize the
/// probability that unfrequently executed code doesn't pollute the cache, by
/// putting clusters in descending order of hotness.
class OptimizeCacheReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit OptimizeCacheReorderAlgorithm(
      std::unique_ptr<ClusterAlgorithm> CAlgo)
      : ReorderAlgorithm(std::move(CAlgo)) {}

  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

/// A new reordering algorithm for basic blocks, ext-tsp
class ExtTSPReorderAlgorithm : public ReorderAlgorithm {
public:
  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

/// Toy example that simply reverses the original basic block order.
class ReverseReorderAlgorithm : public ReorderAlgorithm {
public:
  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

/// Create clusters as usual and place them in random order.
class RandomClusterReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit RandomClusterReorderAlgorithm(
      std::unique_ptr<ClusterAlgorithm> CAlgo)
      : ReorderAlgorithm(std::move(CAlgo)) {}

  void reorderBasicBlocks(BinaryFunction &BF,
                          BasicBlockOrder &Order) const override;
};

} // namespace bolt
} // namespace llvm

#endif
