//===- bolt/Passes/CallGraph.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_BINARY_FUNCTION_CALLGRAPH_H
#define BOLT_PASSES_BINARY_FUNCTION_CALLGRAPH_H

#include "bolt/Passes/CallGraph.h"
#include <deque>
#include <functional>
#include <unordered_map>

namespace llvm {
namespace bolt {

class BinaryFunction;
class BinaryContext;

class BinaryFunctionCallGraph : public CallGraph {
public:
  NodeId maybeGetNodeId(const BinaryFunction *BF) const {
    auto Itr = FuncToNodeId.find(BF);
    return Itr != FuncToNodeId.end() ? Itr->second : InvalidId;
  }
  NodeId getNodeId(const BinaryFunction *BF) const {
    auto Itr = FuncToNodeId.find(BF);
    assert(Itr != FuncToNodeId.end());
    return Itr->second;
  }
  BinaryFunction *nodeIdToFunc(NodeId Id) {
    assert(Id < Funcs.size());
    return Funcs[Id];
  }
  const BinaryFunction *nodeIdToFunc(NodeId Id) const {
    assert(Id < Funcs.size());
    return Funcs[Id];
  }
  NodeId addNode(BinaryFunction *BF, uint32_t Size, uint64_t Samples = 0);

  /// Compute a DFS traversal of the call graph.
  std::deque<BinaryFunction *> buildTraversalOrder();

private:
  std::unordered_map<const BinaryFunction *, NodeId> FuncToNodeId;
  std::vector<BinaryFunction *> Funcs;
};

using CgFilterFunction = std::function<bool(const BinaryFunction &BF)>;
inline bool NoFilter(const BinaryFunction &) { return false; }

/// Builds a call graph from the map of BinaryFunctions provided in BC.
/// The arguments control how the graph is constructed.
/// Filter is called on each function, any function that it returns true for
/// is omitted from the graph.
/// If IncludeSplitCalls is true, then calls from cold BBs are considered for
/// the graph, otherwise they are ignored. UseFunctionHotSize controls whether
/// the hot size of a function is used when filling in the Size attribute of new
/// Nodes. UseEdgeCounts is used to control if the Weight attribute on Arcs is
/// computed using the number of calls.
BinaryFunctionCallGraph
buildCallGraph(BinaryContext &BC, CgFilterFunction Filter = NoFilter,
               bool CgFromPerfData = false, bool IncludeSplitCalls = true,
               bool UseFunctionHotSize = false, bool UseSplitHotSize = false,
               bool UseEdgeCounts = false, bool IgnoreRecursiveCalls = false);

} // namespace bolt
} // namespace llvm

#endif
