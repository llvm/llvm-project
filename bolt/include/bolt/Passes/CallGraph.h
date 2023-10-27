//===- bolt/Passes/CallGraph.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CALLGRAPH_H
#define BOLT_PASSES_CALLGRAPH_H

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace llvm {
namespace bolt {

// TODO: find better place for this
inline int64_t hashCombine(const int64_t Seed, const int64_t Val) {
  std::hash<int64_t> Hasher;
  return Seed ^ (Hasher(Val) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2));
}

/// A call graph class.
class CallGraph {
public:
  using NodeId = size_t;
  static constexpr NodeId InvalidId = -1;

  template <typename T> class iterator_range {
    T Begin;
    T End;

  public:
    template <typename Container>
    iterator_range(Container &&c) : Begin(c.begin()), End(c.end()) {}
    iterator_range(T Begin, T End)
        : Begin(std::move(Begin)), End(std::move(End)) {}

    T begin() const { return Begin; }
    T end() const { return End; }
  };

  class Arc {
  public:
    struct Hash {
      int64_t operator()(const Arc &Arc) const;
    };

    Arc(NodeId S, NodeId D, double W = 0) : Src(S), Dst(D), Weight(W) {}
    Arc(const Arc &) = delete;

    friend bool operator==(const Arc &Lhs, const Arc &Rhs) {
      return Lhs.Src == Rhs.Src && Lhs.Dst == Rhs.Dst;
    }

    NodeId src() const { return Src; }
    NodeId dst() const { return Dst; }
    double weight() const { return Weight; }
    double avgCallOffset() const { return AvgCallOffset; }
    double normalizedWeight() const { return NormalizedWeight; }

  private:
    friend class CallGraph;
    NodeId Src{InvalidId};
    NodeId Dst{InvalidId};
    mutable double Weight{0};
    mutable double NormalizedWeight{0};
    mutable double AvgCallOffset{0};
  };

  using ArcsType = std::unordered_set<Arc, Arc::Hash>;
  using ArcIterator = ArcsType::iterator;
  using ArcConstIterator = ArcsType::const_iterator;

  class Node {
  public:
    explicit Node(uint32_t Size, uint64_t Samples = 0)
        : Size(Size), Samples(Samples) {}

    uint32_t size() const { return Size; }
    uint64_t samples() const { return Samples; }

    const std::vector<NodeId> &successors() const { return Succs; }
    const std::vector<NodeId> &predecessors() const { return Preds; }

  private:
    friend class CallGraph;
    uint32_t Size;
    uint64_t Samples;

    // preds and succs contain no duplicate elements and self arcs are not
    // allowed
    std::vector<NodeId> Preds;
    std::vector<NodeId> Succs;
  };

  size_t numNodes() const { return Nodes.size(); }
  size_t numArcs() const { return Arcs.size(); }
  const Node &getNode(const NodeId Id) const {
    assert(Id < Nodes.size());
    return Nodes[Id];
  }
  uint32_t size(const NodeId Id) const {
    assert(Id < Nodes.size());
    return Nodes[Id].Size;
  }
  uint64_t samples(const NodeId Id) const {
    assert(Id < Nodes.size());
    return Nodes[Id].Samples;
  }
  const std::vector<NodeId> &successors(const NodeId Id) const {
    assert(Id < Nodes.size());
    return Nodes[Id].Succs;
  }
  const std::vector<NodeId> &predecessors(const NodeId Id) const {
    assert(Id < Nodes.size());
    return Nodes[Id].Preds;
  }
  NodeId addNode(uint32_t Size, uint64_t Samples = 0);
  const Arc &incArcWeight(NodeId Src, NodeId Dst, double W = 1.0,
                          double Offset = 0.0);
  ArcIterator findArc(NodeId Src, NodeId Dst) {
    return Arcs.find(Arc(Src, Dst));
  }
  ArcConstIterator findArc(NodeId Src, NodeId Dst) const {
    return Arcs.find(Arc(Src, Dst));
  }
  iterator_range<ArcConstIterator> arcs() const {
    return iterator_range<ArcConstIterator>(Arcs.begin(), Arcs.end());
  }
  iterator_range<std::vector<Node>::const_iterator> nodes() const {
    return iterator_range<std::vector<Node>::const_iterator>(Nodes.begin(),
                                                             Nodes.end());
  }

  double density() const {
    return double(Arcs.size()) / (Nodes.size() * Nodes.size());
  }

  // Initialize NormalizedWeight field for every arc
  void normalizeArcWeights();
  // Make sure that the sum of incoming arc weights is at least the number of
  // samples for every node
  void adjustArcWeights();

  template <typename L> void printDot(StringRef FileName, L getLabel) const;

private:
  void setSamples(const NodeId Id, uint64_t Samples) {
    assert(Id < Nodes.size());
    Nodes[Id].Samples = Samples;
  }

  std::vector<Node> Nodes;
  ArcsType Arcs;
};

template <class L>
void CallGraph::printDot(StringRef FileName, L GetLabel) const {
  std::error_code EC;
  raw_fd_ostream OS(FileName, EC, sys::fs::OF_None);
  if (EC)
    return;

  OS << "digraph g {\n";
  for (NodeId F = 0; F < Nodes.size(); F++) {
    if (Nodes[F].samples() == 0)
      continue;
    OS << "f" << F << " [label=\"" << GetLabel(F)
       << "\\nsamples=" << Nodes[F].samples() << "\\nsize=" << Nodes[F].size()
       << "\"];\n";
  }
  for (NodeId F = 0; F < Nodes.size(); F++) {
    if (Nodes[F].samples() == 0)
      continue;
    for (NodeId Dst : Nodes[F].successors()) {
      ArcConstIterator Arc = findArc(F, Dst);
      OS << "f" << F << " -> f" << Dst
         << " [label=\"normWgt=" << format("%.3lf", Arc->normalizedWeight())
         << ",weight=" << format("%.0lf", Arc->weight())
         << ",callOffset=" << format("%.1lf", Arc->avgCallOffset()) << "\"];\n";
    }
  }
  OS << "}\n";
}

} // namespace bolt
} // namespace llvm

#endif
