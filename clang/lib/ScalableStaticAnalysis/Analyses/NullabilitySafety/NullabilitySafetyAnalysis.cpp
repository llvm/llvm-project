//===- NullabilitySafetyAnalysis.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/NullabilitySafety/NullabilitySafetyAnalysis.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysis/Analyses/NullabilitySafety/NullabilitySafety.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowFormat.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <memory>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;

namespace {

Expected<EntityPointerLevelSet>
readSet(const json::Object &Obj, StringRef Key,
        JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content = Obj.getArray(Key);
  if (!Content)
    return makeSawButExpectedError(Obj, "an object with a key %s",
                                   Key.str().c_str());
  return entityPointerLevelSetFromJSON(*Content, IdFromJSON);
}

json::Array writeSet(const EntityPointerLevelSet &EPLs,
                     JSONFormat::EntityIdToJSONFn IdToJSON) {
  return entityPointerLevelSetToJSON(make_range(EPLs.begin(), EPLs.end()),
                                     IdToJSON);
}

json::Object serializeEvidence(const NullabilityEvidenceAnalysisResult &R,
                               JSONFormat::EntityIdToJSONFn IdToJSON) {
  return json::Object{
      {"NonnullEvidence", writeSet(R.NonnullEvidence, IdToJSON)},
      {"NullableEvidence", writeSet(R.NullableEvidence, IdToJSON)},
      {"AllReturnsNonnull", writeSet(R.AllReturnsNonnull, IdToJSON)},
      {"MaybeNullEvidence", writeSet(R.MaybeNullEvidence, IdToJSON)},
      {"UnknownEvidence", writeSet(R.UnknownEvidence, IdToJSON)},
      {"ConditionalEvidence",
       edgeSetToJSON(make_range(R.ConditionalEvidence.begin(),
                                R.ConditionalEvidence.end()),
                     IdToJSON)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeEvidence(const json::Object &Obj,
                    JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  auto Ret = std::make_unique<NullabilityEvidenceAnalysisResult>();
  for (auto [Key, Set] :
       {std::pair{"NonnullEvidence", &Ret->NonnullEvidence},
        std::pair{"NullableEvidence", &Ret->NullableEvidence},
        std::pair{"AllReturnsNonnull", &Ret->AllReturnsNonnull},
        std::pair{"MaybeNullEvidence", &Ret->MaybeNullEvidence},
        std::pair{"UnknownEvidence", &Ret->UnknownEvidence}}) {
    Expected<EntityPointerLevelSet> EPLs = readSet(Obj, Key, IdFromJSON);
    if (!EPLs)
      return EPLs.takeError();
    *Set = std::move(*EPLs);
  }
  const json::Array *Conditional = Obj.getArray("ConditionalEvidence");
  if (!Conditional)
    return makeSawButExpectedError(Obj, "an object with a key %s",
                                   "ConditionalEvidence");
  Expected<EdgeSet> Edges = edgeSetFromJSON(*Conditional, IdFromJSON);
  if (!Edges)
    return Edges.takeError();
  Ret->ConditionalEvidence = std::move(*Edges);
  return std::move(Ret);
}

JSONFormat::AnalysisResultRegistry::Add<NullabilityEvidenceAnalysisResult>
    RegisterNullabilityEvidenceResultForJSON(serializeEvidence,
                                             deserializeEvidence);

class NullabilityEvidenceAnalysis final
    : public SummaryAnalysis<NullabilityEvidenceAnalysisResult,
                             NullabilitySafetyEntitySummary> {
public:
  Error add(EntityId, const NullabilitySafetyEntitySummary &Summary) override {
    NullabilityEvidenceAnalysisResult &R = getResult();
    R.NonnullEvidence.insert(Summary.getNonnullEvidence().begin(),
                             Summary.getNonnullEvidence().end());
    R.NullableEvidence.insert(Summary.getNullableEvidence().begin(),
                              Summary.getNullableEvidence().end());
    R.AllReturnsNonnull.insert(Summary.getAllReturnsNonnull().begin(),
                               Summary.getAllReturnsNonnull().end());
    R.MaybeNullEvidence.insert(Summary.getMaybeNullEvidence().begin(),
                               Summary.getMaybeNullEvidence().end());
    R.UnknownEvidence.insert(Summary.getUnknownEvidence().begin(),
                             Summary.getUnknownEvidence().end());
    for (const auto &[Assignee, Sources] : Summary.getConditionalEvidence())
      R.ConditionalEvidence[Assignee].insert(Sources.begin(), Sources.end());
    return Error::success();
  }
};

AnalysisRegistry::Add<NullabilityEvidenceAnalysis>
    RegisterNullabilityEvidenceAnalysis("Whole-program nullability evidence");

json::Object serializeInference(const NullabilityInferenceAnalysisResult &R,
                                JSONFormat::EntityIdToJSONFn IdToJSON) {
  return json::Object{
      {"Nonnull", writeSet(R.Nonnull, IdToJSON)},
      {"Nullable", writeSet(R.Nullable, IdToJSON)},
      {"NullableReachable", writeSet(R.NullableReachable, IdToJSON)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeInference(const json::Object &Obj,
                     JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  auto Ret = std::make_unique<NullabilityInferenceAnalysisResult>();
  for (auto [Key, Set] :
       {std::pair{"Nonnull", &Ret->Nonnull},
        std::pair{"Nullable", &Ret->Nullable},
        std::pair{"NullableReachable", &Ret->NullableReachable}}) {
    Expected<EntityPointerLevelSet> EPLs = readSet(Obj, Key, IdFromJSON);
    if (!EPLs)
      return EPLs.takeError();
    *Set = std::move(*EPLs);
  }
  return std::move(Ret);
}

JSONFormat::AnalysisResultRegistry::Add<NullabilityInferenceAnalysisResult>
    RegisterNullabilityInferenceResultForJSON(serializeInference,
                                              deserializeInference);

class NullabilityInferenceAnalysis final
    : public DerivedAnalysis<NullabilityInferenceAnalysisResult,
                             PointerFlowAnalysisResult,
                             NullabilityEvidenceAnalysisResult> {
  std::map<EntityPointerLevel, std::vector<EntityPointerLevel>> AssigneesOf;
  EntityPointerLevelSet Candidates;
  EntityPointerLevelSet Unknown;
  EdgeSet Conditional;

public:
  Error initialize(const PointerFlowAnalysisResult &PointerFlow,
                   const NullabilityEvidenceAnalysisResult &Evidence) override {
    for (const auto &[Contributor, Edges] : PointerFlow.Edges)
      for (const auto &[Assignee, Values] : Edges)
        for (const EntityPointerLevel &Value : Values)
          AssigneesOf[Value].push_back(Assignee);

    NullabilityInferenceAnalysisResult &R = getResult();
    R.Nullable = Evidence.NullableEvidence;
    R.NullableReachable = Evidence.NullableEvidence;
    R.NullableReachable.insert(Evidence.MaybeNullEvidence.begin(),
                               Evidence.MaybeNullEvidence.end());
    Candidates = Evidence.NonnullEvidence;
    Candidates.insert(Evidence.AllReturnsNonnull.begin(),
                      Evidence.AllReturnsNonnull.end());
    for (const auto &[Assignee, Sources] : Evidence.ConditionalEvidence)
      Candidates.insert(Assignee);
    Unknown = Evidence.UnknownEvidence;
    Conditional = Evidence.ConditionalEvidence;
    return Error::success();
  }

  Expected<bool> step() override {
    NullabilityInferenceAnalysisResult &R = getResult();
    std::vector<EntityPointerLevel> Worklist(R.NullableReachable.begin(),
                                             R.NullableReachable.end());
    while (!Worklist.empty()) {
      EntityPointerLevel Value = Worklist.back();
      Worklist.pop_back();
      auto It = AssigneesOf.find(Value);
      if (It == AssigneesOf.end())
        continue;
      for (const EntityPointerLevel &Assignee : It->second)
        if (R.NullableReachable.insert(Assignee).second)
          Worklist.push_back(Assignee);
    }

    std::map<EntityPointerLevel, unsigned> Pending;
    std::map<EntityPointerLevel, std::vector<EntityPointerLevel>> DependentsOf;
    for (const EntityPointerLevel &EPL : Candidates) {
      if (R.NullableReachable.count(EPL) || Unknown.count(EPL))
        continue;
      auto It = Conditional.find(EPL);
      unsigned N = It == Conditional.end() ? 0 : It->second.size();
      Pending[EPL] = N;
      if (N == 0) {
        R.Nonnull.insert(EPL);
        Worklist.push_back(EPL);
        continue;
      }
      for (const EntityPointerLevel &Source : It->second)
        DependentsOf[Source].push_back(EPL);
    }
    while (!Worklist.empty()) {
      EntityPointerLevel Source = Worklist.back();
      Worklist.pop_back();
      auto It = DependentsOf.find(Source);
      if (It == DependentsOf.end())
        continue;
      for (const EntityPointerLevel &Dependent : It->second)
        if (--Pending[Dependent] == 0 && R.Nonnull.insert(Dependent).second)
          Worklist.push_back(Dependent);
    }
    return false;
  }
};

AnalysisRegistry::Add<NullabilityInferenceAnalysis>
    RegisterNullabilityInferenceAnalysis(
        "Nullability inferred from evidence and pointer flow");

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int NullabilitySafetyAnalysisAnchorSource = 0;
} // namespace clang::ssaf
