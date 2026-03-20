//===- ExamplePlugin.cpp - Example SSAF plugin ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A loadable plugin that demonstrates the full SSAF analysis pipeline.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// TagsEntitySummary
//
// Per-entity data: a list of string tags. Stored in the LU data section
// under summary_name "TagsEntitySummary". Serialized as:
//   { "tags": ["tag1", "tag2", ...] }
//===----------------------------------------------------------------------===//

struct TagsEntitySummary final : EntitySummary {
  static SummaryName summaryName() { return SummaryName("TagsEntitySummary"); }

  SummaryName getSummaryName() const override {
    return SummaryName("TagsEntitySummary");
  }

  std::vector<std::string> Tags;
};

json::Object serializeTagsEntitySummary(const EntitySummary &ES,
                                        JSONFormat::EntityIdToJSONFn) {
  const auto &S = static_cast<const TagsEntitySummary &>(ES);
  json::Array TagsArray;
  for (const auto &Tag : S.Tags) {
    TagsArray.push_back(Tag);
  }
  return json::Object{{"tags", std::move(TagsArray)}};
}

Expected<std::unique_ptr<EntitySummary>>
deserializeTagsEntitySummary(const json::Object &Obj, EntityIdTable &,
                             JSONFormat::EntityIdFromJSONFn) {
  const json::Array *TagsArray = Obj.getArray("tags");
  if (!TagsArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'tags'");
  }

  auto S = std::make_unique<TagsEntitySummary>();
  for (const auto &[Index, Val] : llvm::enumerate(*TagsArray)) {
    auto Str = Val.getAsString();
    if (!Str) {
      return createStringError(inconvertibleErrorCode(),
                               "tags element at index %zu is not a string",
                               Index);
    }
    S->Tags.push_back(Str->str());
  }
  return std::move(S);
}

struct TagsEntitySummaryFormatInfo final : JSONFormat::FormatInfo {
  TagsEntitySummaryFormatInfo()
      : JSONFormat::FormatInfo(SummaryName("TagsEntitySummary"),
                               serializeTagsEntitySummary,
                               deserializeTagsEntitySummary) {}
};

llvm::Registry<JSONFormat::FormatInfo>::Add<TagsEntitySummaryFormatInfo>
    RegisterTagsEntitySummaryForJSON("TagsEntitySummary",
                                     "JSON format info for TagsEntitySummary");

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
// TagsAnalysisResult
//
// Sorted, deduplicated list of all tags seen across entities. Serialized as:
//   { "tags": ["tag1", "tag2", ...] }
//===----------------------------------------------------------------------===//

struct TagsAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("TagsAnalysisResult");
  }

  std::vector<std::string> Tags;
};

json::Object serializeTagsAnalysisResult(const TagsAnalysisResult &R,
                                         JSONFormat::EntityIdToJSONFn) {
  json::Array TagsArray;
  for (const auto &Tag : R.Tags) {
    TagsArray.push_back(Tag);
  }
  return json::Object{{"tags", std::move(TagsArray)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeTagsAnalysisResult(const json::Object &Obj,
                              JSONFormat::EntityIdFromJSONFn) {
  const json::Array *TagsArray = Obj.getArray("tags");
  if (!TagsArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'tags'");
  }

  auto R = std::make_unique<TagsAnalysisResult>();
  for (const auto &[Index, Val] : llvm::enumerate(*TagsArray)) {
    auto S = Val.getAsString();
    if (!S) {
      return createStringError(inconvertibleErrorCode(),
                               "tags element at index %zu is not a string",
                               Index);
    }
    R->Tags.push_back(S->str());
  }
  return std::move(R);
}

JSONFormat::AnalysisResultRegistry::Add<TagsAnalysisResult>
    RegisterTagsResultForJSON(serializeTagsAnalysisResult,
                              deserializeTagsAnalysisResult);

//===----------------------------------------------------------------------===//
// PairsAnalysisResult
//
// Per-entity pair count. Serialized as:
//   { "pair_counts": [{"entity_id": {...}, "count": N}, ...] }
//===----------------------------------------------------------------------===//

struct PairsAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("PairsAnalysisResult");
  }

  std::vector<std::pair<EntityId, int>> PairCounts;
};

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
// TagsPairsAnalysisResult
//
// Aggregate statistics derived from TagsAnalysisResult and
// PairsAnalysisResult. Serialized as:
//   { "unique_tag_count": N, "entity_count": N,
//     "total_pair_count": N, "max_pairs_per_entity": N }
//===----------------------------------------------------------------------===//

struct TagsPairsAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("TagsPairsAnalysisResult");
  }

  int UniqueTagCount = 0;
  int EntityCount = 0;
  int TotalPairCount = 0;
  int MaxPairsPerEntity = 0;
};

json::Object serializeTagsPairsAnalysisResult(const TagsPairsAnalysisResult &R,
                                              JSONFormat::EntityIdToJSONFn) {
  return json::Object{{"unique_tag_count", R.UniqueTagCount},
                      {"entity_count", R.EntityCount},
                      {"total_pair_count", R.TotalPairCount},
                      {"max_pairs_per_entity", R.MaxPairsPerEntity}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeTagsPairsAnalysisResult(const json::Object &Obj,
                                   JSONFormat::EntityIdFromJSONFn) {
  auto R = std::make_unique<TagsPairsAnalysisResult>();

  auto UTC = Obj.getInteger("unique_tag_count");
  if (!UTC) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid 'unique_tag_count'");
  }
  R->UniqueTagCount = static_cast<int>(*UTC);

  auto EC = Obj.getInteger("entity_count");
  if (!EC) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid 'entity_count'");
  }
  R->EntityCount = static_cast<int>(*EC);

  auto TPC = Obj.getInteger("total_pair_count");
  if (!TPC) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid 'total_pair_count'");
  }
  R->TotalPairCount = static_cast<int>(*TPC);

  auto MPE = Obj.getInteger("max_pairs_per_entity");
  if (!MPE) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid 'max_pairs_per_entity'");
  }
  R->MaxPairsPerEntity = static_cast<int>(*MPE);

  return std::move(R);
}

JSONFormat::AnalysisResultRegistry::Add<TagsPairsAnalysisResult>
    RegisterSummaryResultForJSON(serializeTagsPairsAnalysisResult,
                                 deserializeTagsPairsAnalysisResult);

//===----------------------------------------------------------------------===//
// TagsAnalysis
//
// SummaryAnalysis that reads per-entity TagsEntitySummary data and collects
// all unique tags into a sorted, deduplicated flat list.
//===----------------------------------------------------------------------===//

class TagsAnalysis final
    : public SummaryAnalysis<TagsAnalysisResult, TagsEntitySummary> {
public:
  using ResultType = TagsAnalysisResult;

  llvm::Error add(EntityId, const TagsEntitySummary &S) override {
    for (const auto &Tag : S.Tags) {
      result().Tags.push_back(Tag);
    }
    return llvm::Error::success();
  }

  llvm::Error finalize() override {
    auto &Tags = result().Tags;
    std::sort(Tags.begin(), Tags.end());
    Tags.erase(std::unique(Tags.begin(), Tags.end()), Tags.end());
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<TagsAnalysis>
    RegisterTagsAnalysis("Collects unique tags across entities");

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

//===----------------------------------------------------------------------===//
// TagsPairsAnalysis
//
// DerivedAnalysis that depends on TagsAnalysisResult and PairsAnalysisResult.
// Computes aggregate statistics:
//   - unique_tag_count:      number of distinct tags
//   - entity_count:          number of entities with pair data
//   - total_pair_count:      sum of all per-entity pair counts
//   - max_pairs_per_entity:  maximum pairs on any single entity
//===----------------------------------------------------------------------===//

class TagsPairsAnalysis final
    : public DerivedAnalysis<TagsPairsAnalysisResult, TagsAnalysisResult,
                             PairsAnalysisResult> {
public:
  using ResultType = TagsPairsAnalysisResult;

  llvm::Error initialize(const TagsAnalysisResult &Tags,
                         const PairsAnalysisResult &Pairs) override {
    result().UniqueTagCount = static_cast<int>(Tags.Tags.size());
    result().EntityCount = static_cast<int>(Pairs.PairCounts.size());

    int Total = 0;
    int Max = 0;
    for (const auto &[Id, Count] : Pairs.PairCounts) {
      Total += Count;
      if (Count > Max)
        Max = Count;
    }
    result().TotalPairCount = Total;
    result().MaxPairsPerEntity = Max;

    return llvm::Error::success();
  }

  llvm::Expected<bool> step() override {
    return false; // converged in initialize
  }
};

AnalysisRegistry::Add<TagsPairsAnalysis>
    RegisterTagsPairsAnalysis("Aggregate tag and pair statistics");

} // namespace
