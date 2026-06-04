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

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
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
  return Ret;
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
  return Ret;
}

JSONFormat::AnalysisResultRegistry::Add<UnsafeBufferReachableAnalysisResult>
    RegisterUnsafeBufferReachableResultForJSON(
        serializeUnsafeBufferReachableAnalysisResult,
        deserializeUnsafeBufferReachableAnalysisResult);

/// Computes all the reachable "nodes" (pointers) in a pointer flow graph from a
/// provided starter node set.  Specifically, the starter set is the unsafe
/// pointers found by `UnsafeBufferUsageAnalysis`.
class UnsafeBufferReachableAnalysis
    : public DerivedAnalysis<UnsafeBufferReachableAnalysisResult,
                             PointerFlowAnalysisResult,
                             UnsafeBufferUsageAnalysisResult> {
  using GraphT = std::map<EntityId, EdgeSet>;
  const GraphT *Graph = nullptr;

  // Use pointers for efficiency. Both `Graph` and `Reachables` in the result
  // are tree-based containers that only grow. So pointers to them are stable.
  using EPLPtr = const EntityPointerLevel *;

  // Find all outgoing edges from `EPL` in the `Graph`, insert their
  // destination nodes into `Reachables`, and add newly discovered nodes to
  // `Worklist`:
  void updateReachablesWithOutgoings(EPLPtr EPL,
                                     std::vector<EPLPtr> &WorkList) {
    for (auto &[Id, SubGraph] : *Graph) {
      auto I = SubGraph.find(*EPL);
      EntityPointerLevelSet &ReachablesOfId = getResult().Reachables[Id];

      if (I != SubGraph.end()) {
        for (const auto &EPL : I->second) {
          auto [Ignored, Inserted] = ReachablesOfId.insert(EPL);
          if (Inserted)
            WorkList.push_back(&EPL);
        }
      }
    }
  }

public:
  llvm::Error
  initialize(const PointerFlowAnalysisResult &Graph,
             const UnsafeBufferUsageAnalysisResult &Starter) override {
    this->Graph = &Graph.Edges;
    assert(getResult().Reachables.empty());
    getResult().Reachables.insert(Starter.begin(), Starter.end());
    return llvm::Error::success();
  }

  llvm::Expected<bool> step() override {
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
    // This is not an iterative algorithm so stop iteration by retruning false:
    return false;
  }
};

AnalysisRegistry::Add<UnsafeBufferReachableAnalysis>
    RegisterUnsafeBufferReachableAnalysis(
        "Reachable pointers from unsafe buffer usage in pointer flow graph");

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageAnalysisAnchorSource = 0;
