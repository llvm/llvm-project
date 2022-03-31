//===- bolt/Passes/HFSortPlus.cpp - Order functions by hotness ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// hfsort+ - layout of hot functions with i-TLB cache optimization.
//
// Given an ordering of hot functions (and hence, their assignment to the
// i-TLB pages), we can divide all functions calls Into two categories:
// - 'short' ones that have a caller-callee distance less than a page;
// - 'long' ones where the distance exceeds a page.
// The short calls are likely to result in a i-TLB cache hit. For the long ones,
// the hit/miss result depends on the 'hotness' of the page (i.e., how often
// the page is accessed). Assuming that functions are sent to the i-TLB cache
// in a random order, the probability that a page is present in the cache is
// proportional to the number of samples corresponding to the functions on the
// page. The following algorithm detects short and long calls, and optimizes
// the expected number of cache misses for the long ones.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/HFSort.h"
#include "llvm/Support/CommandLine.h"
#include <cmath>
#include <set>
#include <vector>

#define DEBUG_TYPE "hfsort"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<unsigned>
ITLBPageSize("itlb-page-size",
  cl::desc("The size of i-tlb cache page"),
  cl::init(4096),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
ITLBEntries("itlb-entries",
  cl::desc("The number of entries in i-tlb cache"),
  cl::init(16),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
ITLBDensity("itlb-density",
  cl::desc("The density of i-tlb cache"),
  cl::init(4096),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<double>
MergeProbability("merge-probability",
  cl::desc("The minimum probability of a call for merging two clusters"),
  cl::init(0.9),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<double>
ArcThreshold("arc-threshold",
  cl::desc("The threshold for ignoring arcs with a small relative weight"),
  cl::init(0.00000001),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;

namespace {

class Edge;
using ArcList = std::vector<const Arc *>;

// A chain (ordered sequence) of nodes (functions) in the call graph
class Chain {
public:
  Chain(const Chain &) = delete;
  Chain(Chain &&) = default;
  Chain &operator=(const Chain &) = delete;
  Chain &operator=(Chain &&) = default;

  explicit Chain(size_t Id_, NodeId Node, size_t Samples_, size_t Size_)
      : Id(Id_), Samples(Samples_), Size(Size_), Nodes(1, Node) {}

  double density() const { return static_cast<double>(Samples) / Size; }

  Edge *getEdge(Chain *Other) const {
    for (std::pair<Chain *, Edge *> It : Edges)
      if (It.first == Other)
        return It.second;
    return nullptr;
  }

  void removeEdge(Chain *Other) {
    auto It = Edges.begin();
    while (It != Edges.end()) {
      if (It->first == Other) {
        Edges.erase(It);
        return;
      }
      It++;
    }
  }

  void addEdge(Chain *Other, Edge *Edge) { Edges.emplace_back(Other, Edge); }

  void merge(Chain *Other) {
    Nodes.insert(Nodes.end(), Other->Nodes.begin(), Other->Nodes.end());
    Samples += Other->Samples;
    Size += Other->Size;
  }

  void mergeEdges(Chain *Other);

  void clear() {
    Nodes.clear();
    Edges.clear();
  }

public:
  size_t Id;
  uint64_t Samples;
  uint64_t Size;
  // Cached score for the chain
  double Score{0};
  // Cached short-calls for the chain
  double ShortCalls{0};
  // Nodes in the chain
  std::vector<NodeId> Nodes;
  // Adjacent chains and corresponding edges (lists of arcs)
  std::vector<std::pair<Chain *, Edge *>> Edges;
};

// An edge in the call graph representing Arcs between two Chains.
// When functions are merged Into chains, the edges are combined too so that
// there is always at most one edge between a pair of chains
class Edge {
public:
  Edge(const Edge &) = delete;
  Edge(Edge &&) = default;
  Edge &operator=(const Edge &) = delete;
  Edge &operator=(Edge &&) = default;

  explicit Edge(Chain *SrcChain_, Chain *DstChain_, const Arc *A)
      : SrcChain(SrcChain_), DstChain(DstChain_), Arcs(1, A) {}

  void changeEndpoint(Chain *From, Chain *To) {
    if (From == SrcChain)
      SrcChain = To;
    if (From == DstChain)
      DstChain = To;
  }

  void moveArcs(Edge *Other) {
    Arcs.insert(Arcs.end(), Other->Arcs.begin(), Other->Arcs.end());
    Other->Arcs.clear();
  }

  void setMergeGain(Chain *PredChain, double ForwardGain, double BackwardGain) {
    // When forward and backward gains are the same, prioritize merging that
    // preserves the original order of the functions in the binary
    if (std::abs(ForwardGain - BackwardGain) < 1e-8) {
      if (SrcChain->Id < DstChain->Id) {
        IsGainForward = true;
        CachedGain = PredChain == SrcChain ? ForwardGain : BackwardGain;
      } else {
        IsGainForward = false;
        CachedGain = PredChain == SrcChain ? BackwardGain : ForwardGain;
      }
    } else if (ForwardGain > BackwardGain) {
      IsGainForward = PredChain == SrcChain;
      CachedGain = ForwardGain;
    } else {
      IsGainForward = PredChain != SrcChain;
      CachedGain = BackwardGain;
    }
  }

  double gain() const { return CachedGain; }

  Chain *predChain() const { return IsGainForward ? SrcChain : DstChain; }

  Chain *succChain() const { return IsGainForward ? DstChain : SrcChain; }

private:
  Chain *SrcChain{nullptr};
  Chain *DstChain{nullptr};

public:
  // Original arcs in the binary with corresponding execution counts
  ArcList Arcs;
  // Cached gain of merging the pair of chains
  double CachedGain{-1.0};
  // Since the gain of merging (Src, Dst) and (Dst, Src) might be different,
  // we store a flag indicating which of the options results in a higher gain
  bool IsGainForward;
};

void Chain::mergeEdges(Chain *Other) {
  // Update edges adjacent to chain other
  for (auto EdgeIt : Other->Edges) {
    Chain *const DstChain = EdgeIt.first;
    Edge *const DstEdge = EdgeIt.second;
    Chain *const TargetChain = DstChain == Other ? this : DstChain;

    // Find the corresponding edge in the current chain
    Edge *CurEdge = getEdge(TargetChain);
    if (CurEdge == nullptr) {
      DstEdge->changeEndpoint(Other, this);
      this->addEdge(TargetChain, DstEdge);
      if (DstChain != this && DstChain != Other)
        DstChain->addEdge(this, DstEdge);
    } else {
      CurEdge->moveArcs(DstEdge);
    }
    // Cleanup leftover edge
    if (DstChain != Other)
      DstChain->removeEdge(Other);
  }
}

class HFSortPlus {
public:
  explicit HFSortPlus(const CallGraph &Cg) : Cg(Cg) { initialize(); }

  /// Run the algorithm and return ordered set of function clusters.
  std::vector<Cluster> run() {
    // Pass 1
    runPassOne();

    // Pass 2
    runPassTwo();

    outs() << "BOLT-INFO: hfsort+ reduced the number of chains from "
           << Cg.numNodes() << " to " << HotChains.size() << "\n";

    // Sorting chains by density in decreasing order
    auto DensityComparator = [](const Chain *L, const Chain *R) {
      if (L->density() != R->density())
        return L->density() > R->density();
      // Making sure the comparison is deterministic
      return L->Id < R->Id;
    };
    std::stable_sort(HotChains.begin(), HotChains.end(), DensityComparator);

    // Return the set of clusters that are left, which are the ones that
    // didn't get merged (so their first func is its original func)
    std::vector<Cluster> Clusters;
    Clusters.reserve(HotChains.size());
    for (Chain *Chain : HotChains)
      Clusters.emplace_back(Cluster(Chain->Nodes, Cg));
    return Clusters;
  }

private:
  /// Initialize the set of active chains, function id to chain mapping,
  /// total number of samples and function addresses.
  void initialize() {
    OutWeight.resize(Cg.numNodes(), 0);
    InWeight.resize(Cg.numNodes(), 0);
    AllChains.reserve(Cg.numNodes());
    HotChains.reserve(Cg.numNodes());
    NodeChain.resize(Cg.numNodes(), nullptr);
    Addr.resize(Cg.numNodes(), 0);

    // Initialize chains
    for (NodeId F = 0; F < Cg.numNodes(); ++F) {
      AllChains.emplace_back(F, F, Cg.samples(F), Cg.size(F));
      HotChains.push_back(&AllChains.back());
      NodeChain[F] = &AllChains.back();
      TotalSamples += Cg.samples(F);
      for (NodeId Succ : Cg.successors(F)) {
        if (F == Succ)
          continue;
        const Arc &Arc = *Cg.findArc(F, Succ);
        OutWeight[F] += Arc.weight();
        InWeight[Succ] += Arc.weight();
      }
    }

    AllEdges.reserve(Cg.numArcs());
    for (NodeId F = 0; F < Cg.numNodes(); ++F) {
      for (NodeId Succ : Cg.successors(F)) {
        if (F == Succ)
          continue;
        const Arc &Arc = *Cg.findArc(F, Succ);
        if (Arc.weight() == 0.0 ||
            Arc.weight() / TotalSamples < opts::ArcThreshold) {
          continue;
        }

        Edge *CurEdge = NodeChain[F]->getEdge(NodeChain[Succ]);
        if (CurEdge != nullptr) {
          // This edge is already present in the graph
          assert(NodeChain[Succ]->getEdge(NodeChain[F]) != nullptr);
          CurEdge->Arcs.push_back(&Arc);
        } else {
          // This is a new edge
          AllEdges.emplace_back(NodeChain[F], NodeChain[Succ], &Arc);
          NodeChain[F]->addEdge(NodeChain[Succ], &AllEdges.back());
          NodeChain[Succ]->addEdge(NodeChain[F], &AllEdges.back());
        }
      }
    }

    for (Chain *&Chain : HotChains) {
      Chain->ShortCalls = shortCalls(Chain);
      Chain->Score = score(Chain);
    }
  }

  /// The probability that a page with a given density is not in the cache.
  ///
  /// Assume that the hot functions are called in a random order; then the
  /// probability of an i-TLB page being accessed after a function call is
  /// p = pageSamples / TotalSamples. The probability that the page is not
  /// accessed is (1 - p), and the probability that it is not in the cache
  /// (i.e. not accessed during the last kCacheEntries function calls)
  /// is (1 - p)^kCacheEntries
  double missProbability(double ChainDensity) const {
    double PageSamples = ChainDensity * opts::ITLBDensity;

    if (PageSamples >= TotalSamples)
      return 0;

    double P = PageSamples / TotalSamples;
    return pow(1.0 - P, double(opts::ITLBEntries));
  }

  /// The expected number of calls on different i-TLB pages for an arc of the
  /// call graph with a specified weight
  double expectedCalls(uint64_t SrcAddr, uint64_t DstAddr,
                       double Weight) const {
    uint64_t Dist = SrcAddr >= DstAddr ? SrcAddr - DstAddr : DstAddr - SrcAddr;
    if (Dist >= opts::ITLBPageSize)
      return 0;

    double D = double(Dist) / double(opts::ITLBPageSize);
    // Increasing the importance of shorter calls
    return (1.0 - D * D) * Weight;
  }

  /// The expected number of calls within a given chain with both endpoints on
  /// the same cache page
  double shortCalls(Chain *Chain) const {
    Edge *Edge = Chain->getEdge(Chain);
    if (Edge == nullptr)
      return 0;

    double Calls = 0;
    for (const Arc *Arc : Edge->Arcs) {
      uint64_t SrcAddr = Addr[Arc->src()] + uint64_t(Arc->avgCallOffset());
      uint64_t DstAddr = Addr[Arc->dst()];
      Calls += expectedCalls(SrcAddr, DstAddr, Arc->weight());
    }
    return Calls;
  }

  /// The number of calls between the two chains with both endpoints on
  /// the same i-TLB page, assuming that a given pair of chains gets merged
  double shortCalls(Chain *ChainPred, Chain *ChainSucc, Edge *Edge) const {
    double Calls = 0;
    for (const Arc *Arc : Edge->Arcs) {
      Chain *SrcChain = NodeChain[Arc->src()];
      uint64_t SrcAddr;
      uint64_t DstAddr;
      if (SrcChain == ChainPred) {
        SrcAddr = Addr[Arc->src()] + uint64_t(Arc->avgCallOffset());
        DstAddr = Addr[Arc->dst()] + ChainPred->Size;
      } else {
        SrcAddr =
            Addr[Arc->src()] + uint64_t(Arc->avgCallOffset()) + ChainPred->Size;
        DstAddr = Addr[Arc->dst()];
      }
      Calls += expectedCalls(SrcAddr, DstAddr, Arc->weight());
    }

    Calls += ChainPred->ShortCalls;
    Calls += ChainSucc->ShortCalls;

    return Calls;
  }

  double score(Chain *Chain) const {
    double LongCalls = Chain->Samples - Chain->ShortCalls;
    return LongCalls * missProbability(Chain->density());
  }

  /// The gain of merging two chains.
  ///
  /// We assume that the final chains are sorted by their density, and hence
  /// every chain is likely to be adjacent with chains of the same density.
  /// Thus, the 'hotness' of every chain can be estimated by density*pageSize,
  /// which is used to compute the probability of cache misses for long calls
  /// of a given chain.
  /// The result is also scaled by the size of the resulting chain in order to
  /// increase the chance of merging short chains, which is helpful for
  /// the i-cache performance.
  double mergeGain(Chain *ChainPred, Chain *ChainSucc, Edge *Edge) const {
    // Cache misses on the chains before merging
    double CurScore = ChainPred->Score + ChainSucc->Score;

    // Cache misses on the merged chain
    double LongCalls = ChainPred->Samples + ChainSucc->Samples -
                       shortCalls(ChainPred, ChainSucc, Edge);
    const double MergedSamples = ChainPred->Samples + ChainSucc->Samples;
    const double MergedSize = ChainPred->Size + ChainSucc->Size;
    double NewScore = LongCalls * missProbability(MergedSamples / MergedSize);

    double Gain = CurScore - NewScore;
    // Scale the result to increase the importance of merging short chains
    Gain /= std::min(ChainPred->Size, ChainSucc->Size);

    return Gain;
  }

  /// Run the first optimization pass of the algorithm:
  /// Merge chains that call each other with a high probability.
  void runPassOne() {
    // Find candidate pairs of chains for merging
    std::vector<const Arc *> ArcsToMerge;
    for (Chain *ChainPred : HotChains) {
      NodeId F = ChainPred->Nodes.back();
      for (NodeId Succ : Cg.successors(F)) {
        if (F == Succ)
          continue;

        const Arc &Arc = *Cg.findArc(F, Succ);
        if (Arc.weight() == 0.0 ||
            Arc.weight() / TotalSamples < opts::ArcThreshold)
          continue;

        const double CallsFromPred = OutWeight[F];
        const double CallsToSucc = InWeight[Succ];
        const double CallsPredSucc = Arc.weight();

        // Probability that the first chain is calling the second one
        const double ProbOut =
            CallsFromPred > 0 ? CallsPredSucc / CallsFromPred : 0;
        assert(0.0 <= ProbOut && ProbOut <= 1.0 && "incorrect out-probability");

        // Probability that the second chain is called From the first one
        const double ProbIn = CallsToSucc > 0 ? CallsPredSucc / CallsToSucc : 0;
        assert(0.0 <= ProbIn && ProbIn <= 1.0 && "incorrect in-probability");

        if (std::min(ProbOut, ProbIn) >= opts::MergeProbability)
          ArcsToMerge.push_back(&Arc);
      }
    }

    // Sort the pairs by the weight in reverse order
    std::sort(
        ArcsToMerge.begin(), ArcsToMerge.end(),
        [](const Arc *L, const Arc *R) { return L->weight() > R->weight(); });

    // Merge the pairs of chains
    for (const Arc *Arc : ArcsToMerge) {
      Chain *ChainPred = NodeChain[Arc->src()];
      Chain *ChainSucc = NodeChain[Arc->dst()];
      if (ChainPred == ChainSucc)
        continue;
      if (ChainPred->Nodes.back() == Arc->src() &&
          ChainSucc->Nodes.front() == Arc->dst())
        mergeChains(ChainPred, ChainSucc);
    }
  }

  /// Run the second optimization pass of the hfsort+ algorithm:
  /// Merge pairs of chains while there is an improvement in the
  /// expected cache miss ratio.
  void runPassTwo() {
    // Creating a priority queue containing all edges ordered by the merge gain
    auto GainComparator = [](Edge *L, Edge *R) {
      if (std::abs(L->gain() - R->gain()) > 1e-8)
        return L->gain() > R->gain();

      // Making sure the comparison is deterministic
      if (L->predChain()->Id != R->predChain()->Id)
        return L->predChain()->Id < R->predChain()->Id;

      return L->succChain()->Id < R->succChain()->Id;
    };
    std::set<Edge *, decltype(GainComparator)> Queue(GainComparator);

    // Inserting the edges Into the queue
    for (Chain *ChainPred : HotChains) {
      for (auto EdgeIt : ChainPred->Edges) {
        Chain *ChainSucc = EdgeIt.first;
        Edge *ChainEdge = EdgeIt.second;
        // Ignore loop edges
        if (ChainPred == ChainSucc)
          continue;
        // Ignore already processed edges
        if (ChainEdge->gain() != -1.0)
          continue;

        // Compute the gain of merging the two chains
        auto ForwardGain = mergeGain(ChainPred, ChainSucc, ChainEdge);
        auto BackwardGain = mergeGain(ChainSucc, ChainPred, ChainEdge);
        ChainEdge->setMergeGain(ChainPred, ForwardGain, BackwardGain);
        if (ChainEdge->gain() > 0.0)
          Queue.insert(ChainEdge);
      }
    }

    // Merge the chains while the gain of merging is positive
    while (!Queue.empty()) {
      // Extract the best (top) edge for merging
      Edge *It = *Queue.begin();
      Queue.erase(Queue.begin());
      Edge *BestEdge = It;
      Chain *BestChainPred = BestEdge->predChain();
      Chain *BestChainSucc = BestEdge->succChain();
      if (BestChainPred == BestChainSucc || BestEdge->gain() <= 0.0)
        continue;

      // Remove outdated edges
      for (std::pair<Chain *, Edge *> EdgeIt : BestChainPred->Edges)
        Queue.erase(EdgeIt.second);
      for (std::pair<Chain *, Edge *> EdgeIt : BestChainSucc->Edges)
        Queue.erase(EdgeIt.second);

      // Merge the best pair of chains
      mergeChains(BestChainPred, BestChainSucc);

      // Insert newly created edges Into the queue
      for (auto EdgeIt : BestChainPred->Edges) {
        Chain *ChainSucc = EdgeIt.first;
        Edge *ChainEdge = EdgeIt.second;
        // Ignore loop edges
        if (BestChainPred == ChainSucc)
          continue;

        // Compute the gain of merging the two chains
        auto ForwardGain = mergeGain(BestChainPred, ChainSucc, ChainEdge);
        auto BackwardGain = mergeGain(ChainSucc, BestChainPred, ChainEdge);
        ChainEdge->setMergeGain(BestChainPred, ForwardGain, BackwardGain);
        if (ChainEdge->gain() > 0.0)
          Queue.insert(ChainEdge);
      }
    }
  }

  /// Merge chain From into chain Into and update the list of active chains.
  void mergeChains(Chain *Into, Chain *From) {
    assert(Into != From && "cannot merge a chain with itself");
    Into->merge(From);

    // Update the chains and addresses for functions merged from From
    size_t CurAddr = 0;
    for (NodeId F : Into->Nodes) {
      NodeChain[F] = Into;
      Addr[F] = CurAddr;
      CurAddr += Cg.size(F);
    }

    // Merge edges
    Into->mergeEdges(From);
    From->clear();

    // Update cached scores for the new chain
    Into->ShortCalls = shortCalls(Into);
    Into->Score = score(Into);

    // Remove chain From From the list of active chains
    auto it = std::remove(HotChains.begin(), HotChains.end(), From);
    HotChains.erase(it, HotChains.end());
  }

private:
  // The call graph
  const CallGraph &Cg;

  // All chains of functions
  std::vector<Chain> AllChains;

  // Active chains. The vector gets updated at runtime when chains are merged
  std::vector<Chain *> HotChains;

  // All edges between chains
  std::vector<Edge> AllEdges;

  // Node_id => chain
  std::vector<Chain *> NodeChain;

  // Current address of the function From the beginning of its chain
  std::vector<uint64_t> Addr;

  // Total weight of outgoing arcs for each function
  std::vector<double> OutWeight;

  // Total weight of incoming arcs for each function
  std::vector<double> InWeight;
  // The total number of samples in the graph
  double TotalSamples{0};
};

} // end anonymous namespace

std::vector<Cluster> hfsortPlus(CallGraph &Cg) {
  // It is required that the sum of incoming arc weights is not greater
  // than the number of samples for every function.
  // Ensuring the call graph obeys the property before running the algorithm.
  Cg.adjustArcWeights();
  return HFSortPlus(Cg).run();
}

} // namespace bolt
} // namespace llvm
