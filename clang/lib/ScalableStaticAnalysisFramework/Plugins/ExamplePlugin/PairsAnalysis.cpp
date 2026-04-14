//===- PairsAnalysis.cpp - Pairs analysis for ExamplePlugin ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisResults.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <utility>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;
using example_plugin::PairsAnalysisResult;

namespace {

//===----------------------------------------------------------------------===//
// PairsEntitySummary
//
// Per-entity data: a list of (EntityId, EntityId) pairs. Stored in the LU
// data section under summary_name "PairsEntitySummary". Serialized as:
//   { "pairs": [{"first": {...}, "second": {...}}, ...] }
//===----------------------------------------------------------------------===//

struct PairsEntitySummary final : EntitySummary {
  static SummaryName summaryName() { return SummaryName("PairsEntitySummary"); }

  SummaryName getSummaryName() const override {
    return SummaryName("PairsEntitySummary");
  }

  std::vector<std::pair<EntityId, EntityId>> Pairs;
};

json::Object serializePairsEntitySummary(const EntitySummary &ES,
                                         JSONFormat::EntityIdToJSONFn ToJSON) {
  const auto &S = static_cast<const PairsEntitySummary &>(ES);
  json::Array PairsArray;
  for (const auto &[First, Second] : S.Pairs) {
    PairsArray.push_back(json::Object{
        {"first", ToJSON(First)},
        {"second", ToJSON(Second)},
    });
  }
  return json::Object{{"pairs", std::move(PairsArray)}};
}

Expected<std::unique_ptr<EntitySummary>>
deserializePairsEntitySummary(const json::Object &Obj, EntityIdTable &,
                              JSONFormat::EntityIdFromJSONFn FromJSON) {
  auto Result = std::make_unique<PairsEntitySummary>();
  const json::Array *PairsArray = Obj.getArray("pairs");
  if (!PairsArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'pairs'");
  }
  for (const auto &[Index, Value] : llvm::enumerate(*PairsArray)) {
    const json::Object *Pair = Value.getAsObject();
    if (!Pair) {
      return createStringError(
          inconvertibleErrorCode(),
          "pairs element at index %zu is not a JSON object", Index);
    }
    const json::Object *FirstObj = Pair->getObject("first");
    if (!FirstObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'first' field at index '%zu'", Index);
    }
    const json::Object *SecondObj = Pair->getObject("second");
    if (!SecondObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'second' field at index '%zu'", Index);
    }
    auto ExpectedFirst = FromJSON(*FirstObj);
    if (!ExpectedFirst) {
      return createStringError(inconvertibleErrorCode(),
                               "invalid 'first' entity id at index '%zu': %s",
                               Index,
                               toString(ExpectedFirst.takeError()).c_str());
    }
    auto ExpectedSecond = FromJSON(*SecondObj);
    if (!ExpectedSecond) {
      return createStringError(inconvertibleErrorCode(),
                               "invalid 'second' entity id at index '%zu': %s",
                               Index,
                               toString(ExpectedSecond.takeError()).c_str());
    }
    Result->Pairs.emplace_back(*ExpectedFirst, *ExpectedSecond);
  }
  return std::move(Result);
}

struct PairsEntitySummaryFormatInfo final : JSONFormat::FormatInfo {
  PairsEntitySummaryFormatInfo()
      : JSONFormat::FormatInfo(SummaryName("PairsEntitySummary"),
                               serializePairsEntitySummary,
                               deserializePairsEntitySummary) {}
};

llvm::Registry<JSONFormat::FormatInfo>::Add<PairsEntitySummaryFormatInfo>
    RegisterPairsEntitySummaryForJSON(
        "PairsEntitySummary", "JSON format info for PairsEntitySummary");

//===----------------------------------------------------------------------===//
// PairsAnalysisResult serialization
//
// Per-entity pair count. Serialized as:
//   { "pair_counts": [{"entity_id": {...}, "count": N}, ...] }
//===----------------------------------------------------------------------===//

json::Object serializePairsAnalysisResult(const PairsAnalysisResult &R,
                                          JSONFormat::EntityIdToJSONFn ToJSON) {
  json::Array Arr;
  for (const auto &[EI, Count] : R.PairCounts) {
    Arr.push_back(json::Object{{"entity_id", ToJSON(EI)}, {"count", Count}});
  }
  return json::Object{{"pair_counts", std::move(Arr)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializePairsAnalysisResult(const json::Object &Obj,
                               JSONFormat::EntityIdFromJSONFn FromJSON) {
  const json::Array *Arr = Obj.getArray("pair_counts");
  if (!Arr) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'pair_counts'");
  }

  auto R = std::make_unique<PairsAnalysisResult>();
  for (const auto &[Index, Val] : llvm::enumerate(*Arr)) {
    const json::Object *Entry = Val.getAsObject();
    if (!Entry) {
      return createStringError(
          inconvertibleErrorCode(),
          "pair_counts element at index %zu is not an object", Index);
    }
    const json::Object *EIObj = Entry->getObject("entity_id");
    if (!EIObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'entity_id' field at index %zu", Index);
    }
    auto ExpectedEI = FromJSON(*EIObj);
    if (!ExpectedEI) {
      return ExpectedEI.takeError();
    }

    auto CountVal = Entry->getInteger("count");
    if (!CountVal) {
      return createStringError(inconvertibleErrorCode(),
                               "missing or invalid 'count' field at index %zu",
                               Index);
    }
    R->PairCounts.emplace_back(*ExpectedEI, static_cast<int>(*CountVal));
  }
  return std::move(R);
}

JSONFormat::AnalysisResultRegistry::Add<PairsAnalysisResult>
    RegisterPairsResultForJSON(serializePairsAnalysisResult,
                               deserializePairsAnalysisResult);

//===----------------------------------------------------------------------===//
// PairsAnalysis
//
// SummaryAnalysis that reads per-entity PairsEntitySummary data and counts
// the number of pairs per entity, producing (EntityId, count) pairs.
//===----------------------------------------------------------------------===//

class PairsAnalysis final
    : public SummaryAnalysis<PairsAnalysisResult, PairsEntitySummary> {
public:
  using ResultType = PairsAnalysisResult;

  llvm::Error add(EntityId Id, const PairsEntitySummary &S) override {
    result().PairCounts.emplace_back(Id, static_cast<int>(S.Pairs.size()));
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<PairsAnalysis>
    RegisterPairsAnalysis("Counts pairs per entity");

} // namespace
