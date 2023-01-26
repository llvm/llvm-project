//===- bolt/Passes/CallGraph.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CallGraph class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/CallGraph.h"

#define DEBUG_TYPE "callgraph"

#if defined(__x86_64__) && !defined(_MSC_VER)
#  if (!defined USE_SSECRC)
#    define USE_SSECRC
#  endif
#else
#  undef USE_SSECRC
#endif

namespace {

LLVM_ATTRIBUTE_UNUSED inline size_t hash_int64_fallback(int64_t k) {
  uint64_t key = (unsigned long long)k;
  // "64 bit Mix Functions", from Thomas Wang's "Integer Hash Function."
  // http://www.concentric.net/~ttwang/tech/inthash.htm
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  return static_cast<size_t>(static_cast<uint32_t>(key));
}

LLVM_ATTRIBUTE_UNUSED inline size_t hash_int64(int64_t k) {
#if defined(USE_SSECRC) && defined(__SSE4_2__)
  size_t h = 0;
  __asm("crc32q %1, %0\n" : "+r"(h) : "rm"(k));
  return h;
#else
  return hash_int64_fallback(k);
#endif
}

inline size_t hash_int64_pair(int64_t k1, int64_t k2) {
#if defined(USE_SSECRC) && defined(__SSE4_2__)
  // crc32 is commutative, so we need to perturb k1 so that (k1, k2) hashes
  // differently from (k2, k1).
  k1 += k1;
  __asm("crc32q %1, %0\n" : "+r" (k1) : "rm"(k2));
  return k1;
#else
  return (hash_int64(k1) << 1) ^ hash_int64(k2);
#endif
}

}

namespace llvm {
namespace bolt {

int64_t CallGraph::Arc::Hash::operator()(const Arc &Arc) const {
#ifdef USE_STD_HASH
  std::hash<int64_t> Hasher;
  return hashCombine(Hasher(Arc.src()), Arc.dst());
#else
  return hash_int64_pair(int64_t(Arc.src()), int64_t(Arc.dst()));
#endif
}

CallGraph::NodeId CallGraph::addNode(uint32_t Size, uint64_t Samples) {
  NodeId Id = Nodes.size();
  Nodes.emplace_back(Size, Samples);
  return Id;
}

const CallGraph::Arc &CallGraph::incArcWeight(NodeId Src, NodeId Dst, double W,
                                              double Offset) {
  assert(Offset <= size(Src) && "Call offset exceeds function size");

  std::pair<ArcIterator, bool> Res = Arcs.emplace(Src, Dst, W);
  if (!Res.second) {
    Res.first->Weight += W;
    Res.first->AvgCallOffset += Offset * W;
    return *Res.first;
  }
  Res.first->AvgCallOffset = Offset * W;
  Nodes[Src].Succs.push_back(Dst);
  Nodes[Dst].Preds.push_back(Src);
  return *Res.first;
}

void CallGraph::normalizeArcWeights() {
  for (NodeId FuncId = 0; FuncId < numNodes(); ++FuncId) {
    const Node &Func = getNode(FuncId);
    for (NodeId Caller : Func.predecessors()) {
      ArcIterator Arc = findArc(Caller, FuncId);
      Arc->NormalizedWeight = Arc->weight() / Func.samples();
      if (Arc->weight() > 0)
        Arc->AvgCallOffset /= Arc->weight();
      assert(Arc->AvgCallOffset <= size(Caller) &&
             "Avg call offset exceeds function size");
    }
  }
}

void CallGraph::adjustArcWeights() {
  for (NodeId FuncId = 0; FuncId < numNodes(); ++FuncId) {
    const Node &Func = getNode(FuncId);
    uint64_t InWeight = 0;
    for (NodeId Caller : Func.predecessors()) {
      ArcIterator Arc = findArc(Caller, FuncId);
      InWeight += (uint64_t)Arc->weight();
    }
    if (Func.samples() < InWeight)
      setSamples(FuncId, InWeight);
  }
}

}
}
