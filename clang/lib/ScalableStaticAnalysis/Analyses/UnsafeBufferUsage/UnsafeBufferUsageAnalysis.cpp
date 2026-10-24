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
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
/// 4. **C4 (Family):** If a parameter or return slot of a virtual method is
///    unsafe at some pointer level, so is every slot of its override family
///    (provided by `VirtualMethodFamilyAnalysisResult`) at that level, because
///    a virtual call can dispatch to any of the overrides.
class UnsafeBufferReachableAnalysis
    : public DerivedAnalysis<
          UnsafeBufferReachableAnalysisResult, PointerFlowAnalysisResult,
          TypeConstrainedPointersAnalysisResult,
          UnsafeBufferUsageAnalysisResult, VirtualMethodFamilyAnalysisResult> {

  /// The pointer flow graph, partitioned by contributor.
  const std::map<EntityId, EdgeSet> *PointerFlows = nullptr;

  /// The type-constrained entities, which are never unsafe.
  const TypeConstrainedPointersAnalysisResult *TypeConstraints = nullptr;

  /// The C1 unsafe pointers, partitioned by contributor.
  const UnsafeBufferUsageAnalysisResult *UnsafePtrs = nullptr;

  /// Maps each virtual method slot to the ID of its override family.
  const llvm::DenseMap<EntityId, EntityId> *FamilyOf = nullptr;

  /// The slots of each override family.
  llvm::DenseMap<EntityId, llvm::SmallVector<EntityId, 2>> FamilyMembers;

  // Use pointers for efficiency. EPLs are in tree-based containers that only
  // grow. So pointers to them are stable.
  using EPLPtr = const EntityPointerLevel *;

  // Insert `EPL` into `Reachables`, and add it to `Worklist` if it is new.
  // Type-constrained pointers are never inserted, so the search never passes
  // through them (C3):
  void insertReachable(const EntityPointerLevel &EPL,
                       std::vector<EPLPtr> &WorkList) {
    if (TypeConstraints->contains(EPL.getEntity()))
      return;
    auto [It, Inserted] = getResult().Reachables.insert(EPL);
    if (Inserted)
      WorkList.push_back(&*It);
  }

  // Find all outgoing edges from `EPL` in the pointer flow graph, insert their
  // destination nodes into `Reachables`, and add newly discovered nodes to
  // `Worklist`:
  void updateReachablesWithOutgoings(EPLPtr EPL,
                                     std::vector<EPLPtr> &WorkList) {
    for (const EdgeSet &SubGraph : llvm::make_second_range(*PointerFlows)) {
      auto I = SubGraph.find(*EPL);
      if (I == SubGraph.end())
        continue;
      for (const auto &Dst : I->second)
        insertReachable(Dst, WorkList);
    }
  }

  // Insert the slots of the override family of `EPL` at the pointer level of
  // `EPL` into `Reachables`, and add newly discovered nodes to `Worklist`:
  void updateReachablesWithFamily(EPLPtr EPL, std::vector<EPLPtr> &WorkList) {
    auto FamilyIt = FamilyOf->find(EPL->getEntity());
    if (FamilyIt == FamilyOf->end())
      return;
    auto MembersIt = FamilyMembers.find(FamilyIt->second);
    if (MembersIt == FamilyMembers.end())
      return;
    for (EntityId Member : MembersIt->second)
      insertReachable(buildEntityPointerLevel(Member, EPL->getPointerLevel()),
                      WorkList);
  }

  // Compute all pointers reachable from the C1 pointers into
  // `getResult().Reachables`, satisfying C1, C2, C3 and C4.
  void computeReachableUnsafePointers() {
    // Simple DFS:
    std::vector<EPLPtr> Worklist;

    for (const auto &EPLs : llvm::make_second_range(*UnsafePtrs))
      for (const auto &EPL : EPLs)
        insertReachable(EPL, Worklist);

    while (!Worklist.empty()) {
      EPLPtr Node = Worklist.back();
      Worklist.pop_back();

      updateReachablesWithOutgoings(Node, Worklist);
      updateReachablesWithFamily(Node, Worklist);
    }
  }

public:
  llvm::Error
  initialize(const PointerFlowAnalysisResult &PtrFlowGraph,
             const TypeConstrainedPointersAnalysisResult &TypeConstraints,
             const UnsafeBufferUsageAnalysisResult &UnsafePtrs,
             const VirtualMethodFamilyAnalysisResult &Families) override {
    this->PointerFlows = &PtrFlowGraph.Edges;
    this->TypeConstraints = &TypeConstraints;
    this->UnsafePtrs = &UnsafePtrs;
    FamilyOf = &Families.RetAndParamData;
    for (auto [Slot, FamilyId] : Families.RetAndParamData)
      FamilyMembers[FamilyId].push_back(Slot);
    return llvm::Error::success();
  }

  llvm::Expected<bool> step() override {
    // Compute the reachable EPLs from the C1 unsafe pointers over the
    // pointer-flow graph and the override families, skipping type-constrained
    // pointers, so the result satisfies C1, C2, C3, and C4.
    computeReachableUnsafePointers();
    // This is not an iterative algorithm so stop iteration by retruning false:
    return false;
  }
};

AnalysisRegistry::Add<UnsafeBufferReachableAnalysis>
    RegisterUnsafeBufferReachableAnalysis(
        "Reachable pointers from unsafe buffer usage in pointer flow graph, "
        "family-closed across virtual method overrides");

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageAnalysisAnchorSource = 0;
} // namespace clang::ssaf
