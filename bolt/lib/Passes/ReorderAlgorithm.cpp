//===- bolt/Passes/ReorderAlgorithm.cpp - Basic block reordering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used by several basic block reordering
// algorithms.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ReorderAlgorithm.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/CommandLine.h"
#include <queue>
#include <random>
#include <stack>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> NoThreads;

static cl::opt<unsigned> ColdThreshold(
    "cold-threshold",
    cl::desc("tenths of percents of main entry frequency to use as a "
             "threshold when evaluating whether a basic block is cold "
             "(0 means it is only considered cold if the block has zero "
             "samples). Default: 0 "),
    cl::init(0), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> PrintClusters("print-clusters", cl::desc("print clusters"),
                                   cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<uint32_t> RandomSeed("bolt-seed", cl::desc("seed for randomization"),
                             cl::init(42), cl::Hidden,
                             cl::cat(BoltOptCategory));

} // namespace opts

namespace {

template <class T> inline void hashCombine(size_t &Seed, const T &Val) {
  std::hash<T> Hasher;
  Seed ^= Hasher(Val) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2);
}

template <typename A, typename B> struct HashPair {
  size_t operator()(const std::pair<A, B> &Val) const {
    std::hash<A> Hasher;
    size_t Seed = Hasher(Val.first);
    hashCombine(Seed, Val.second);
    return Seed;
  }
};

} // namespace

void ClusterAlgorithm::computeClusterAverageFrequency(const BinaryContext &BC) {
  // Create a separate MCCodeEmitter to allow lock-free execution
  BinaryContext::IndependentCodeEmitter Emitter;
  if (!opts::NoThreads)
    Emitter = BC.createIndependentMCCodeEmitter();

  AvgFreq.resize(Clusters.size(), 0.0);
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I) {
    double Freq = 0.0;
    uint64_t ClusterSize = 0;
    for (BinaryBasicBlock *BB : Clusters[I]) {
      if (BB->getNumNonPseudos() > 0) {
        Freq += BB->getExecutionCount();
        // Estimate the size of a block in bytes at run time
        // NOTE: This might be inaccurate
        ClusterSize += BB->estimateSize(Emitter.MCE.get());
      }
    }
    AvgFreq[I] = ClusterSize == 0 ? 0 : Freq / ClusterSize;
  }
}

void ClusterAlgorithm::printClusters() const {
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I) {
    errs() << "Cluster number " << I;
    if (AvgFreq.size() == Clusters.size())
      errs() << " (frequency: " << AvgFreq[I] << ")";
    errs() << " : ";
    const char *Sep = "";
    for (BinaryBasicBlock *BB : Clusters[I]) {
      errs() << Sep << BB->getName();
      Sep = ", ";
    }
    errs() << "\n";
  }
}

void ClusterAlgorithm::reset() {
  Clusters.clear();
  ClusterEdges.clear();
  AvgFreq.clear();
}

void GreedyClusterAlgorithm::EdgeTy::print(raw_ostream &OS) const {
  OS << Src->getName() << " -> " << Dst->getName() << ", count: " << Count;
}

size_t GreedyClusterAlgorithm::EdgeHash::operator()(const EdgeTy &E) const {
  HashPair<const BinaryBasicBlock *, const BinaryBasicBlock *> Hasher;
  return Hasher(std::make_pair(E.Src, E.Dst));
}

bool GreedyClusterAlgorithm::EdgeEqual::operator()(const EdgeTy &A,
                                                   const EdgeTy &B) const {
  return A.Src == B.Src && A.Dst == B.Dst;
}

void GreedyClusterAlgorithm::clusterBasicBlocks(const BinaryFunction &BF,
                                                bool ComputeEdges) {
  reset();

  // Greedy heuristic implementation for the TSP, applied to BB layout. Try to
  // maximize weight during a path traversing all BBs. In this way, we will
  // convert the hottest branches into fall-throughs.

  // This is the queue of edges from which we will pop edges and use them to
  // cluster basic blocks in a greedy fashion.
  std::vector<EdgeTy> Queue;

  // Initialize inter-cluster weights.
  if (ComputeEdges)
    ClusterEdges.resize(BF.getLayout().block_size());

  // Initialize clusters and edge queue.
  for (BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    // Create a cluster for this BB.
    uint32_t I = Clusters.size();
    Clusters.emplace_back();
    std::vector<BinaryBasicBlock *> &Cluster = Clusters.back();
    Cluster.push_back(BB);
    BBToClusterMap[BB] = I;
    // Populate priority queue with edges.
    auto BI = BB->branch_info_begin();
    for (BinaryBasicBlock *&I : BB->successors()) {
      assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "attempted reordering blocks of function with no profile data");
      Queue.emplace_back(EdgeTy(BB, I, BI->Count));
      ++BI;
    }
  }
  // Sort and adjust the edge queue.
  initQueue(Queue, BF);

  // Grow clusters in a greedy fashion.
  while (!Queue.empty()) {
    EdgeTy E = Queue.back();
    Queue.pop_back();

    const BinaryBasicBlock *SrcBB = E.Src;
    const BinaryBasicBlock *DstBB = E.Dst;

    LLVM_DEBUG(dbgs() << "Popped edge "; E.print(dbgs()); dbgs() << "\n");

    // Case 1: BBSrc and BBDst are the same. Ignore this edge
    if (SrcBB == DstBB || DstBB == *BF.getLayout().block_begin()) {
      LLVM_DEBUG(dbgs() << "\tIgnored (same src, dst)\n");
      continue;
    }

    int I = BBToClusterMap[SrcBB];
    int J = BBToClusterMap[DstBB];

    // Case 2: If they are already allocated at the same cluster, just increase
    // the weight of this cluster
    if (I == J) {
      if (ComputeEdges)
        ClusterEdges[I][I] += E.Count;
      LLVM_DEBUG(dbgs() << "\tIgnored (src, dst belong to the same cluster)\n");
      continue;
    }

    std::vector<BinaryBasicBlock *> &ClusterA = Clusters[I];
    std::vector<BinaryBasicBlock *> &ClusterB = Clusters[J];
    if (areClustersCompatible(ClusterA, ClusterB, E)) {
      // Case 3: SrcBB is at the end of a cluster and DstBB is at the start,
      // allowing us to merge two clusters.
      for (BinaryBasicBlock *BB : ClusterB)
        BBToClusterMap[BB] = I;
      ClusterA.insert(ClusterA.end(), ClusterB.begin(), ClusterB.end());
      ClusterB.clear();
      if (ComputeEdges) {
        // Increase the intra-cluster edge count of cluster A with the count of
        // this edge as well as with the total count of previously visited edges
        // from cluster B cluster A.
        ClusterEdges[I][I] += E.Count;
        ClusterEdges[I][I] += ClusterEdges[J][I];
        // Iterate through all inter-cluster edges and transfer edges targeting
        // cluster B to cluster A.
        for (uint32_t K = 0, E = ClusterEdges.size(); K != E; ++K)
          ClusterEdges[K][I] += ClusterEdges[K][J];
      }
      // Adjust the weights of the remaining edges and re-sort the queue.
      adjustQueue(Queue, BF);
      LLVM_DEBUG(dbgs() << "\tMerged clusters of src, dst\n");
    } else {
      // Case 4: Both SrcBB and DstBB are allocated in positions we cannot
      // merge them. Add the count of this edge to the inter-cluster edge count
      // between clusters A and B to help us decide ordering between these
      // clusters.
      if (ComputeEdges)
        ClusterEdges[I][J] += E.Count;
      LLVM_DEBUG(
          dbgs() << "\tIgnored (src, dst belong to incompatible clusters)\n");
    }
  }
}

void GreedyClusterAlgorithm::reset() {
  ClusterAlgorithm::reset();
  BBToClusterMap.clear();
}

void PHGreedyClusterAlgorithm::initQueue(std::vector<EdgeTy> &Queue,
                                         const BinaryFunction &BF) {
  // Define a comparison function to establish SWO between edges.
  auto Comp = [&BF](const EdgeTy &A, const EdgeTy &B) {
    // With equal weights, prioritize branches with lower index
    // source/destination. This helps to keep original block order for blocks
    // when optimal order cannot be deducted from a profile.
    if (A.Count == B.Count) {
      const signed SrcOrder = BF.getOriginalLayoutRelativeOrder(A.Src, B.Src);
      return (SrcOrder != 0)
                 ? SrcOrder > 0
                 : BF.getOriginalLayoutRelativeOrder(A.Dst, B.Dst) > 0;
    }
    return A.Count < B.Count;
  };

  // Sort edges in increasing profile count order.
  llvm::sort(Queue, Comp);
}

void PHGreedyClusterAlgorithm::adjustQueue(std::vector<EdgeTy> &Queue,
                                           const BinaryFunction &BF) {
  // Nothing to do.
}

bool PHGreedyClusterAlgorithm::areClustersCompatible(const ClusterTy &Front,
                                                     const ClusterTy &Back,
                                                     const EdgeTy &E) const {
  return Front.back() == E.Src && Back.front() == E.Dst;
}

int64_t MinBranchGreedyClusterAlgorithm::calculateWeight(
    const EdgeTy &E, const BinaryFunction &BF) const {
  const BinaryBasicBlock *SrcBB = E.Src;
  const BinaryBasicBlock *DstBB = E.Dst;

  // Initial weight value.
  int64_t W = (int64_t)E.Count;

  // Adjust the weight by taking into account other edges with the same source.
  auto BI = SrcBB->branch_info_begin();
  for (const BinaryBasicBlock *SuccBB : SrcBB->successors()) {
    assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
           "attempted reordering blocks of function with no profile data");
    assert(BI->Count <= std::numeric_limits<int64_t>::max() &&
           "overflow detected");
    // Ignore edges with same source and destination, edges that target the
    // entry block as well as the edge E itself.
    if (SuccBB != SrcBB && SuccBB != *BF.getLayout().block_begin() &&
        SuccBB != DstBB)
      W -= (int64_t)BI->Count;
    ++BI;
  }

  // Adjust the weight by taking into account other edges with the same
  // destination.
  for (const BinaryBasicBlock *PredBB : DstBB->predecessors()) {
    // Ignore edges with same source and destination as well as the edge E
    // itself.
    if (PredBB == DstBB || PredBB == SrcBB)
      continue;
    auto BI = PredBB->branch_info_begin();
    for (const BinaryBasicBlock *SuccBB : PredBB->successors()) {
      if (SuccBB == DstBB)
        break;
      ++BI;
    }
    assert(BI != PredBB->branch_info_end() && "invalid control flow graph");
    assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
           "attempted reordering blocks of function with no profile data");
    assert(BI->Count <= std::numeric_limits<int64_t>::max() &&
           "overflow detected");
    W -= (int64_t)BI->Count;
  }

  return W;
}

void MinBranchGreedyClusterAlgorithm::initQueue(std::vector<EdgeTy> &Queue,
                                                const BinaryFunction &BF) {
  // Initialize edge weights.
  for (const EdgeTy &E : Queue)
    Weight.emplace(std::make_pair(E, calculateWeight(E, BF)));

  // Sort edges in increasing weight order.
  adjustQueue(Queue, BF);
}

void MinBranchGreedyClusterAlgorithm::adjustQueue(std::vector<EdgeTy> &Queue,
                                                  const BinaryFunction &BF) {
  // Define a comparison function to establish SWO between edges.
  auto Comp = [&](const EdgeTy &A, const EdgeTy &B) {
    // With equal weights, prioritize branches with lower index
    // source/destination. This helps to keep original block order for blocks
    // when optimal order cannot be deduced from a profile.
    if (Weight[A] == Weight[B]) {
      const signed SrcOrder = BF.getOriginalLayoutRelativeOrder(A.Src, B.Src);
      return (SrcOrder != 0)
                 ? SrcOrder > 0
                 : BF.getOriginalLayoutRelativeOrder(A.Dst, B.Dst) > 0;
    }
    return Weight[A] < Weight[B];
  };

  // Iterate through all remaining edges to find edges that have their
  // source and destination in the same cluster.
  std::vector<EdgeTy> NewQueue;
  for (const EdgeTy &E : Queue) {
    const BinaryBasicBlock *SrcBB = E.Src;
    const BinaryBasicBlock *DstBB = E.Dst;

    // Case 1: SrcBB and DstBB are the same or DstBB is the entry block. Ignore
    // this edge.
    if (SrcBB == DstBB || DstBB == *BF.getLayout().block_begin()) {
      LLVM_DEBUG(dbgs() << "\tAdjustment: Ignored edge "; E.print(dbgs());
                 dbgs() << " (same src, dst)\n");
      continue;
    }

    int I = BBToClusterMap[SrcBB];
    int J = BBToClusterMap[DstBB];
    std::vector<BinaryBasicBlock *> &ClusterA = Clusters[I];
    std::vector<BinaryBasicBlock *> &ClusterB = Clusters[J];

    // Case 2: They are already allocated at the same cluster or incompatible
    // clusters. Adjust the weights of edges with the same source or
    // destination, so that this edge has no effect on them any more, and ignore
    // this edge. Also increase the intra- (or inter-) cluster edge count.
    if (I == J || !areClustersCompatible(ClusterA, ClusterB, E)) {
      if (!ClusterEdges.empty())
        ClusterEdges[I][J] += E.Count;
      LLVM_DEBUG(dbgs() << "\tAdjustment: Ignored edge "; E.print(dbgs());
                 dbgs() << " (src, dst belong to same cluster or incompatible "
                           "clusters)\n");
      for (const BinaryBasicBlock *SuccBB : SrcBB->successors()) {
        if (SuccBB == DstBB)
          continue;
        auto WI = Weight.find(EdgeTy(SrcBB, SuccBB, 0));
        assert(WI != Weight.end() && "CFG edge not found in Weight map");
        WI->second += (int64_t)E.Count;
      }
      for (const BinaryBasicBlock *PredBB : DstBB->predecessors()) {
        if (PredBB == SrcBB)
          continue;
        auto WI = Weight.find(EdgeTy(PredBB, DstBB, 0));
        assert(WI != Weight.end() && "CFG edge not found in Weight map");
        WI->second += (int64_t)E.Count;
      }
      continue;
    }

    // Case 3: None of the previous cases is true, so just keep this edge in
    // the queue.
    NewQueue.emplace_back(E);
  }

  // Sort remaining edges in increasing weight order.
  Queue.swap(NewQueue);
  llvm::sort(Queue, Comp);
}

bool MinBranchGreedyClusterAlgorithm::areClustersCompatible(
    const ClusterTy &Front, const ClusterTy &Back, const EdgeTy &E) const {
  return Front.back() == E.Src && Back.front() == E.Dst;
}

void MinBranchGreedyClusterAlgorithm::reset() {
  GreedyClusterAlgorithm::reset();
  Weight.clear();
}

void TSPReorderAlgorithm::reorderBasicBlocks(const BinaryFunction &BF,
                                             BasicBlockOrder &Order) const {
  std::vector<std::vector<uint64_t>> Weight;
  std::vector<BinaryBasicBlock *> IndexToBB;

  const size_t N = BF.getLayout().block_size();
  assert(N <= std::numeric_limits<uint64_t>::digits &&
         "cannot use TSP solution for sizes larger than bits in uint64_t");

  // Populating weight map and index map
  for (BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    BB->setLayoutIndex(IndexToBB.size());
    IndexToBB.push_back(BB);
  }
  Weight.resize(N);
  for (BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    auto BI = BB->branch_info_begin();
    Weight[BB->getLayoutIndex()].resize(N);
    for (BinaryBasicBlock *SuccBB : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        Weight[BB->getLayoutIndex()][SuccBB->getLayoutIndex()] = BI->Count;
      ++BI;
    }
  }

  std::vector<std::vector<int64_t>> DP;
  DP.resize(1 << N);
  for (std::vector<int64_t> &Elmt : DP)
    Elmt.resize(N, -1);

  // Start with the entry basic block being allocated with cost zero
  DP[1][0] = 0;
  // Walk through TSP solutions using a bitmask to represent state (current set
  // of BBs in the layout)
  uint64_t BestSet = 1;
  uint64_t BestLast = 0;
  int64_t BestWeight = 0;
  for (uint64_t Set = 1; Set < (1ULL << N); ++Set) {
    // Traverse each possibility of Last BB visited in this layout
    for (uint64_t Last = 0; Last < N; ++Last) {
      // Case 1: There is no possible layout with this BB as Last
      if (DP[Set][Last] == -1)
        continue;

      // Case 2: There is a layout with this Set and this Last, and we try
      // to expand this set with New
      for (uint64_t New = 1; New < N; ++New) {
        // Case 2a: BB "New" is already in this Set
        if ((Set & (1ULL << New)) != 0)
          continue;

        // Case 2b: BB "New" is not in this set and we add it to this Set and
        // record total weight of this layout with "New" as the last BB.
        uint64_t NewSet = (Set | (1ULL << New));
        if (DP[NewSet][New] == -1)
          DP[NewSet][New] = DP[Set][Last] + (int64_t)Weight[Last][New];
        DP[NewSet][New] = std::max(DP[NewSet][New],
                                   DP[Set][Last] + (int64_t)Weight[Last][New]);

        if (DP[NewSet][New] > BestWeight) {
          BestWeight = DP[NewSet][New];
          BestSet = NewSet;
          BestLast = New;
        }
      }
    }
  }

  // Define final function layout based on layout that maximizes weight
  uint64_t Last = BestLast;
  uint64_t Set = BestSet;
  BitVector Visited;
  Visited.resize(N);
  Visited[Last] = true;
  Order.push_back(IndexToBB[Last]);
  Set = Set & ~(1ULL << Last);
  while (Set != 0) {
    int64_t Best = -1;
    uint64_t NewLast;
    for (uint64_t I = 0; I < N; ++I) {
      if (DP[Set][I] == -1)
        continue;
      int64_t AdjWeight = Weight[I][Last] > 0 ? Weight[I][Last] : 0;
      if (DP[Set][I] + AdjWeight > Best) {
        NewLast = I;
        Best = DP[Set][I] + AdjWeight;
      }
    }
    Last = NewLast;
    Visited[Last] = true;
    Order.push_back(IndexToBB[Last]);
    Set = Set & ~(1ULL << Last);
  }
  std::reverse(Order.begin(), Order.end());

  // Finalize layout with BBs that weren't assigned to the layout using the
  // input layout.
  for (BinaryBasicBlock *BB : BF.getLayout().blocks())
    if (Visited[BB->getLayoutIndex()] == false)
      Order.push_back(BB);
}

void OptimizeReorderAlgorithm::reorderBasicBlocks(
    const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.getLayout().block_empty())
    return;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF);

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Arrange basic blocks according to clusters.
  for (ClusterAlgorithm::ClusterTy &Cluster : CAlgo->Clusters)
    Order.insert(Order.end(), Cluster.begin(), Cluster.end());
}

void OptimizeBranchReorderAlgorithm::reorderBasicBlocks(
    const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.getLayout().block_empty())
    return;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF, /* ComputeEdges = */ true);
  std::vector<ClusterAlgorithm::ClusterTy> &Clusters = CAlgo->Clusters;
  std::vector<std::unordered_map<uint32_t, uint64_t>> &ClusterEdges =
      CAlgo->ClusterEdges;

  // Compute clusters' average frequencies.
  CAlgo->computeClusterAverageFrequency(BF.getBinaryContext());
  std::vector<double> &AvgFreq = CAlgo->AvgFreq;

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Cluster layout order
  std::vector<uint32_t> ClusterOrder;

  // Do a topological sort for clusters, prioritizing frequently-executed BBs
  // during the traversal.
  std::stack<uint32_t> Stack;
  std::vector<uint32_t> Status;
  std::vector<uint32_t> Parent;
  Status.resize(Clusters.size(), 0);
  Parent.resize(Clusters.size(), 0);
  constexpr uint32_t STACKED = 1;
  constexpr uint32_t VISITED = 2;
  Status[0] = STACKED;
  Stack.push(0);
  while (!Stack.empty()) {
    uint32_t I = Stack.top();
    if (!(Status[I] & VISITED)) {
      Status[I] |= VISITED;
      // Order successors by weight
      auto ClusterComp = [&ClusterEdges, I](uint32_t A, uint32_t B) {
        return ClusterEdges[I][A] > ClusterEdges[I][B];
      };
      std::priority_queue<uint32_t, std::vector<uint32_t>,
                          decltype(ClusterComp)>
          SuccQueue(ClusterComp);
      for (std::pair<const uint32_t, uint64_t> &Target : ClusterEdges[I]) {
        if (Target.second > 0 && !(Status[Target.first] & STACKED) &&
            !Clusters[Target.first].empty()) {
          Parent[Target.first] = I;
          Status[Target.first] = STACKED;
          SuccQueue.push(Target.first);
        }
      }
      while (!SuccQueue.empty()) {
        Stack.push(SuccQueue.top());
        SuccQueue.pop();
      }
      continue;
    }
    // Already visited this node
    Stack.pop();
    ClusterOrder.push_back(I);
  }
  std::reverse(ClusterOrder.begin(), ClusterOrder.end());
  // Put unreachable clusters at the end
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
    if (!(Status[I] & VISITED) && !Clusters[I].empty())
      ClusterOrder.push_back(I);

  // Sort nodes with equal precedence
  auto Beg = ClusterOrder.begin();
  // Don't reorder the first cluster, which contains the function entry point
  ++Beg;
  std::stable_sort(Beg, ClusterOrder.end(),
                   [&AvgFreq, &Parent](uint32_t A, uint32_t B) {
                     uint32_t P = Parent[A];
                     while (Parent[P] != 0) {
                       if (Parent[P] == B)
                         return false;
                       P = Parent[P];
                     }
                     P = Parent[B];
                     while (Parent[P] != 0) {
                       if (Parent[P] == A)
                         return true;
                       P = Parent[P];
                     }
                     return AvgFreq[A] > AvgFreq[B];
                   });

  if (opts::PrintClusters) {
    errs() << "New cluster order: ";
    const char *Sep = "";
    for (uint32_t O : ClusterOrder) {
      errs() << Sep << O;
      Sep = ", ";
    }
    errs() << '\n';
  }

  // Arrange basic blocks according to cluster order.
  for (uint32_t ClusterIndex : ClusterOrder) {
    ClusterAlgorithm::ClusterTy &Cluster = Clusters[ClusterIndex];
    Order.insert(Order.end(), Cluster.begin(), Cluster.end());
  }
}

void OptimizeCacheReorderAlgorithm::reorderBasicBlocks(
    const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.getLayout().block_empty())
    return;

  const uint64_t ColdThreshold =
      opts::ColdThreshold *
      (*BF.getLayout().block_begin())->getExecutionCount() / 1000;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF);
  std::vector<ClusterAlgorithm::ClusterTy> &Clusters = CAlgo->Clusters;

  // Compute clusters' average frequencies.
  CAlgo->computeClusterAverageFrequency(BF.getBinaryContext());
  std::vector<double> &AvgFreq = CAlgo->AvgFreq;

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Cluster layout order
  std::vector<uint32_t> ClusterOrder;

  // Order clusters based on average instruction execution frequency
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
    if (!Clusters[I].empty())
      ClusterOrder.push_back(I);
  // Don't reorder the first cluster, which contains the function entry point
  std::stable_sort(
      std::next(ClusterOrder.begin()), ClusterOrder.end(),
      [&AvgFreq](uint32_t A, uint32_t B) { return AvgFreq[A] > AvgFreq[B]; });

  if (opts::PrintClusters) {
    errs() << "New cluster order: ";
    const char *Sep = "";
    for (uint32_t O : ClusterOrder) {
      errs() << Sep << O;
      Sep = ", ";
    }
    errs() << '\n';
  }

  // Arrange basic blocks according to cluster order.
  for (uint32_t ClusterIndex : ClusterOrder) {
    ClusterAlgorithm::ClusterTy &Cluster = Clusters[ClusterIndex];
    Order.insert(Order.end(), Cluster.begin(), Cluster.end());
    // Force zero execution count on clusters that do not meet the cut off
    // specified by --cold-threshold.
    if (AvgFreq[ClusterIndex] < static_cast<double>(ColdThreshold))
      for (BinaryBasicBlock *BBPtr : Cluster)
        BBPtr->setExecutionCount(0);
  }
}

void ReverseReorderAlgorithm::reorderBasicBlocks(const BinaryFunction &BF,
                                                 BasicBlockOrder &Order) const {
  if (BF.getLayout().block_empty())
    return;

  BinaryBasicBlock *FirstBB = *BF.getLayout().block_begin();
  Order.push_back(FirstBB);
  for (auto RLI = BF.getLayout().block_rbegin(); *RLI != FirstBB; ++RLI)
    Order.push_back(*RLI);
}

void RandomClusterReorderAlgorithm::reorderBasicBlocks(
    const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.getLayout().block_empty())
    return;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF);
  std::vector<ClusterAlgorithm::ClusterTy> &Clusters = CAlgo->Clusters;

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Cluster layout order
  std::vector<uint32_t> ClusterOrder;

  // Order clusters based on average instruction execution frequency
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
    if (!Clusters[I].empty())
      ClusterOrder.push_back(I);

  std::shuffle(std::next(ClusterOrder.begin()), ClusterOrder.end(),
               std::default_random_engine(opts::RandomSeed.getValue()));

  if (opts::PrintClusters) {
    errs() << "New cluster order: ";
    const char *Sep = "";
    for (uint32_t O : ClusterOrder) {
      errs() << Sep << O;
      Sep = ", ";
    }
    errs() << '\n';
  }

  // Arrange basic blocks according to cluster order.
  for (uint32_t ClusterIndex : ClusterOrder) {
    ClusterAlgorithm::ClusterTy &Cluster = Clusters[ClusterIndex];
    Order.insert(Order.end(), Cluster.begin(), Cluster.end());
  }
}
