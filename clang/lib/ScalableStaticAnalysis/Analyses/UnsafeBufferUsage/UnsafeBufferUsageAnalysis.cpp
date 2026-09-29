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
// EntityPointerLevelSets.
//
// UnsafeBufferReachableAnalysisResult is a flat set of EntityPointerLevels
// reachable from unsafe buffer usage.
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
      entityPointerLevelSetToJSON(R.Reachables, IdToJSON);
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

  auto Reachables = entityPointerLevelSetFromJSON(*Content, IdFromJSON);

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

  struct BoundsPropagationGraph {
    EdgeSet PointerFlows;

    /// Returns the EntityPointerLevelSet that are reachable from \p Src by
    /// one edge in the BoundsPropagationGraph.
    EntityPointerLevelSet getDestNodes(const EntityPointerLevel &Src) const {
      auto I = PointerFlows.find(Src);
      if (I == PointerFlows.end())
        return {};
      return I->second;
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
        auto [It, Inserted] = getResult().Reachables.insert(Dst);
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

    for (auto &EPL : Reachables)
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
        BPG.try_emplace(Id,
                        BoundsPropagationGraph{std::move(FilteredSubGraph)});
    }

    // Filter out type-constrained pointers from `UnsafePtrs`:
    for (auto &[Contributor, EPLs] : UnsafePtrs) {
      auto FilteredRange = llvm::make_filter_range(EPLs, HasNoTypeConstraint);

      getResult().Reachables.insert(FilteredRange.begin(), FilteredRange.end());
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
