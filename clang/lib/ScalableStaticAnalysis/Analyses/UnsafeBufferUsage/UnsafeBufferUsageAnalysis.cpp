//===- UnsafeBufferUsageAnalysis.cpp - WPA for UnsafeBufferUsage ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UnsafeBufferUsageAnalysis is a noop analysis.
//
// UnsafeBufferUsageAnalysisResult is a map from EntityIds to
// EntityPointerLevelSets
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/TypeConstrainedPointers/TypeConstrainedPointers.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

using namespace clang::ssaf;
using namespace llvm;

namespace {

json::Object serializeUnsafeBufferUsageAnalysisResult(
    const UnsafeBufferUsageAnalysisResult &R,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  json::Object Result;

  Result[UnsafeBufferUsageAnalysisResultName] =
      entityPointerLevelMapToJSON(R.UnsafeBuffers, IdToJSON);
  return Result;
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeUnsafeBufferUsageAnalysisResult(
    const json::Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content =
      Obj.getArray(UnsafeBufferUsageAnalysisResultName);

  if (!Content)
    return makeSawButExpectedError(Obj, "an object with a key %s",
                                   UnsafeBufferUsageAnalysisResultName.data());

  auto UnsafeBuffers = entityPointerLevelMapFromJSON(*Content, IdFromJSON);

  if (!UnsafeBuffers)
    return UnsafeBuffers.takeError();

  auto Ret = std::make_unique<UnsafeBufferUsageAnalysisResult>();

  Ret->UnsafeBuffers = std::move(*UnsafeBuffers);
  return std::move(Ret);
}

JSONFormat::AnalysisResultRegistry::Add<UnsafeBufferUsageAnalysisResult>
    RegisterUnsafeBufferUsageResultForJSON(
        serializeUnsafeBufferUsageAnalysisResult,
        deserializeUnsafeBufferUsageAnalysisResult);

class UnsafeBufferUsageAnalysis final
    : public SummaryAnalysis<UnsafeBufferUsageAnalysisResult,
                             UnsafeBufferUsageEntitySummary> {
public:
  llvm::Error add(EntityId Id,
                  const UnsafeBufferUsageEntitySummary &Summary) override {
    auto UnsafeBuffersOfEntity = getUnsafeBuffers(Summary);

    getResult().UnsafeBuffers[Id] = EntityPointerLevelSet(
        UnsafeBuffersOfEntity.begin(), UnsafeBuffersOfEntity.end());
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<UnsafeBufferUsageAnalysis>
    RegisterUnsafeBufferUsageAnalysis(
        "Whole-program unsafe buffer usage analysis");

//===----------------------------------------------------------------------===//
// UnsafeBufferReachableAnalysis---computes reachable unsafe buffer nodes
//===----------------------------------------------------------------------===//

json::Object serializeUnsafeBufferReachableAnalysisResult(
    const UnsafeBufferReachableAnalysisResult &R,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  json::Object Result;

  Result[UnsafeBufferReachableAnalysisResultName] =
      entityPointerLevelMapToJSON(R.Reachables, IdToJSON);
  return Result;
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeUnsafeBufferReachableAnalysisResult(
    const json::Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content =
      Obj.getArray(UnsafeBufferReachableAnalysisResultName);

  if (!Content)
    return makeSawButExpectedError(
        Obj, "an object with a key %s",
        UnsafeBufferReachableAnalysisResultName.data());

  auto Reachables = entityPointerLevelMapFromJSON(*Content, IdFromJSON);

  if (!Reachables)
    return Reachables.takeError();

  auto Ret = std::make_unique<UnsafeBufferReachableAnalysisResult>();

  Ret->Reachables = std::move(*Reachables);
  return std::move(Ret);
}

JSONFormat::AnalysisResultRegistry::Add<UnsafeBufferReachableAnalysisResult>
    RegisterUnsafeBufferReachableResultForJSON(
        serializeUnsafeBufferReachableAnalysisResult,
        deserializeUnsafeBufferReachableAnalysisResult);

/// \brief Computes pointers (EPLs) that satisfy a specific set of constraints.
///
/// The pointers must satisfy all of the following constraints:
///
/// 1. **C1 (Unsafe):** Any pointer in `UnsafeBufferUsageAnalysisResult`
///    is considered unsafe.
/// 2. **C2 (Reachable):** If a pointer is reachable from an unsafe pointer in
///    the pointer flow graph (provided by `PointerFlowAnalysisResult`), it is
///    also unsafe.
/// 3. **C3 (Constrained):** Type-constrained entities are NOT unsafe.
class UnsafeBufferReachableAnalysis
    : public DerivedAnalysis<UnsafeBufferReachableAnalysisResult,
                             PointerFlowAnalysisResult,
                             TypeConstrainedPointersAnalysisResult,
                             UnsafeBufferUsageAnalysisResult> {

  /// BoundsPropagationGraph adds bounds propagation semantics to the
  /// pointer-flow graph, which represents the set of static pointer assignment
  /// sites collected from the source code. Consider the following example:
  ///
  /// void f(int ***p, int **q) {
  ///   *p = q;
  ///   (**p)[5] = 0;
  /// }
  ///
  /// There is one static pointer assignment thus one pointer-flow edge: (p, 2)
  /// -> (q, 1). In terms of bounds propagation, this assignment implies that if
  /// 'p' at pointer level 2 requires bounds, 'q' at pointer level 1 must also
  /// have them. Furthermore, this relationship propagates to deeper indirection
  /// levels: if 'p' at level 3 requires bounds, so does 'q' at level 2.
  ///
  /// In the example above, `(**p)` requires bounds (due to the array index),
  /// and therefore `*q` must require bounds as well.
  ///
  /// To generalize the idea, the BoundsPropagationGraph is defined as a super
  /// graph of the input pointer-flow graph by:
  ///
  ///   For each edge (src, i) -> (dest, j) in the pointer-flow graph, the
  ///   BoundsPropagationGraph has a finite set of edges
  ///   {(src, i + d) -> (dest, j + d) | 0 <= d < UB}, where UB is an upper
  ///   bound based on the maximum pointer level the pointer type can have.
  struct BoundsPropagationGraph {
  private:
    EdgeSet PointerFlows;

  public:
    BoundsPropagationGraph(EdgeSet PointerFlows)
        : PointerFlows(std::move(PointerFlows)) {}

    /// Returns the EntityPointerLevelSet that are reachable from \p Src by
    /// one edge in the BoundsPropagationGraph.
    EntityPointerLevelSet getDestNodes(const EntityPointerLevel &Src) const {
      unsigned SrcPtrLv = Src.getPointerLevel();
      EntityPointerLevelSet Result;

      for (unsigned P = 1; P <= SrcPtrLv; ++P) {
        auto I = PointerFlows.find(buildEntityPointerLevel(Src.getEntity(), P));

        if (I != PointerFlows.end()) {
          unsigned Delta = SrcPtrLv - P;
          for (const auto &EPL : I->second)
            Result.insert(buildEntityPointerLevel(
                EPL.getEntity(), EPL.getPointerLevel() + Delta));
        }
      }
      return Result;
    }
  };

  std::map<EntityId, BoundsPropagationGraph> BPG;

  // Use pointers for efficiency. EPLs are in tree-based containers that only
  // grow. So pointers to them are stable.
  using EPLPtr = const EntityPointerLevel *;

  // Find all outgoing edges from `EPL` in the `Graph`, insert their
  // destination nodes into `Reachables`, and add newly discovered nodes to
  // `Worklist`:
  void updateReachablesWithOutgoings(EPLPtr EPL,
                                     std::vector<EPLPtr> &WorkList) {
    for (auto &[Id, SubGraph] : BPG) {
      auto R = SubGraph.getDestNodes(*EPL);

      for (const auto &Dst : R) {
        auto [It, Inserted] = getResult().Reachables[Id].insert(Dst);
        if (Inserted)
          WorkList.push_back(&*It);
      }
    }
  }

  // Expand the initial set of C1 pointers in `getResult().Reachables` by
  // computing and appending all reachable pointers, satisfying both C1 and C2.
  void computeReachableUnsafePointers() {
    auto &Reachables = getResult().Reachables;
    // Simple DFS:
    std::vector<EPLPtr> Worklist;

    for (auto &[Id, EPLs] : Reachables)
      for (auto &EPL : EPLs)
        Worklist.push_back(&EPL);

    while (!Worklist.empty()) {
      EPLPtr Node = Worklist.back();
      Worklist.pop_back();

      updateReachablesWithOutgoings(Node, Worklist);
    }
  }

public:
  llvm::Error
  initialize(const PointerFlowAnalysisResult &PtrFlowGraph,
             const TypeConstrainedPointersAnalysisResult &TypeConstraints,
             const UnsafeBufferUsageAnalysisResult &UnsafePtrs) override {
    auto HasNoTypeConstraint =
        [&TypeConstraints](const EntityPointerLevel &EPL) {
          return !TypeConstraints.contains(EPL.getEntity());
        };

    // Filter out edges involving type-constrained pointers from `PtrFlowGraph`:
    for (auto &[Id, SubGraph] : PtrFlowGraph.Edges) {
      EdgeSet FilteredSubGraph;

      for (const auto &[Src, Dsts] : SubGraph) {
        if (TypeConstraints.contains(Src.getEntity()))
          continue;

        auto FilteredDstRange =
            llvm::make_filter_range(Dsts, HasNoTypeConstraint);

        if (!FilteredDstRange.empty())
          FilteredSubGraph[Src].insert(FilteredDstRange.begin(),
                                       FilteredDstRange.end());
      }
      if (!FilteredSubGraph.empty())
        BPG.try_emplace(Id, std::move(FilteredSubGraph));
    }

    // Filter out type-constrained pointers from `UnsafePtrs`:
    for (auto &[Contributor, EPLs] : UnsafePtrs) {
      auto FilteredRange = llvm::make_filter_range(EPLs, HasNoTypeConstraint);

      getResult().Reachables[Contributor].insert(FilteredRange.begin(),
                                                 FilteredRange.end());
    }
    return llvm::Error::success();
  }

  llvm::Expected<bool> step() override {
    // Compute the reachable EPLs from the C1 unsafe pointers over the
    // pointer-flow graph; both are already C3-filtered, so the result
    // satisfies C1, C2, and C3.
    computeReachableUnsafePointers();
    // This is not an iterative algorithm so stop iteration by retruning false:
    return false;
  }
};

AnalysisRegistry::Add<UnsafeBufferReachableAnalysis>
    RegisterUnsafeBufferReachableAnalysis(
        "Reachable pointers from unsafe buffer usage in pointer flow graph");

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageAnalysisAnchorSource = 0;
} // namespace clang::ssaf
