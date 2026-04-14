//===- TagsPairsAnalysis.cpp - Derived analysis for ExamplePlugin ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisResults.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"
#include <memory>

using namespace clang::ssaf;
using namespace llvm;
using example_plugin::PairsAnalysisResult;
using example_plugin::TagsAnalysisResult;

namespace {

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
