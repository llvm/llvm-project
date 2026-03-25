//===- ExamplePlugin.cpp - Example SSAF plugin ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A loadable plugin that registers three analysis result types — Tags, Counts,
// and FailingDeserializer — together with their JSON serializers/deserializers
// and trivial DerivedAnalysis implementations. Used by lit tests for both
// clang-ssaf-analyzer and clang-ssaf-format.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// TagsAnalysisResult
//
// Holds a flat list of string tags. Serialized as:
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
    RegisterJSONFormatSupportForTagAnalysisResult(
        serializeTagsAnalysisResult, deserializeTagsAnalysisResult);

//===----------------------------------------------------------------------===//
// TagsAnalysis
//
// A trivial DerivedAnalysis that produces a fixed set of tags, independent
// of the input LUSummary. Used so that lit tests have predictable output.
//===----------------------------------------------------------------------===//

class TagsAnalysis final : public DerivedAnalysis<TagsAnalysisResult> {
public:
  llvm::Error initialize() override { return llvm::Error::success(); }

  llvm::Expected<bool> step() override {
    result().Tags = {"alpha", "beta", "gamma"};
    return false; // converged after one step
  }
};

AnalysisRegistry::Add<TagsAnalysis> RegisterTagsAnalysis(
    "Produces a fixed list of string tags for testing WPASuite serialization");

//===----------------------------------------------------------------------===//
// CountsAnalysisResult
//
// Holds a list of (EntityId, int) count pairs. Serialized as:
//   { "counts": [{"entity_id": {...}, "count": N}, ...] }
//===----------------------------------------------------------------------===//

struct CountsAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("CountsAnalysisResult");
  }

  std::vector<std::pair<EntityId, int>> Counts;
};

json::Object
serializeCountsAnalysisResult(const CountsAnalysisResult &R,
                              JSONFormat::EntityIdToJSONFn ToJSON) {
  json::Array CountsArray;
  for (const auto &[EI, Count] : R.Counts) {
    CountsArray.push_back(
        json::Object{{"entity_id", ToJSON(EI)}, {"count", Count}});
  }
  return json::Object{{"counts", std::move(CountsArray)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeCountsAnalysisResult(const json::Object &Obj,
                                JSONFormat::EntityIdFromJSONFn FromJSON) {
  const json::Array *CountsArray = Obj.getArray("counts");
  if (!CountsArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'counts'");
  }

  auto R = std::make_unique<CountsAnalysisResult>();
  for (const auto &[Index, Val] : llvm::enumerate(*CountsArray)) {
    const json::Object *Entry = Val.getAsObject();
    if (!Entry) {
      return createStringError(inconvertibleErrorCode(),
                               "counts element at index %zu is not an object",
                               Index);
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
    R->Counts.emplace_back(*ExpectedEI, static_cast<int>(*CountVal));
  }
  return std::move(R);
}

JSONFormat::AnalysisResultRegistry::Add<CountsAnalysisResult>
    RegisterCountsForJSON(serializeCountsAnalysisResult,
                          deserializeCountsAnalysisResult);

//===----------------------------------------------------------------------===//
// CountsAnalysis
//
// A trivial DerivedAnalysis that produces an empty counts result. Entity ID
// serialization is exercised by round-trip tests that supply pre-built JSON.
//===----------------------------------------------------------------------===//

class CountsAnalysis final : public DerivedAnalysis<CountsAnalysisResult> {
public:
  llvm::Error initialize() override { return llvm::Error::success(); }

  llvm::Expected<bool> step() override {
    // Produces no counts; entity-ID round-trip is covered by direct JSON tests.
    return false;
  }
};

AnalysisRegistry::Add<CountsAnalysis> RegisterCountsAnalysis(
    "Produces an empty counts result for testing WPASuite serialization");

//===----------------------------------------------------------------------===//
// FailingDeserializerAnalysisResult
//
// An analysis result whose deserializer always returns an error. Used to
// exercise error propagation in the WPASuite read path without needing a
// malformed payload.
//===----------------------------------------------------------------------===//

struct FailingDeserializerAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("FailingDeserializerAnalysisResult");
  }
};

json::Object serializeFailingDeserializerAnalysisResult(
    const FailingDeserializerAnalysisResult &, JSONFormat::EntityIdToJSONFn) {
  return json::Object{};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeFailingDeserializerAnalysisResult(const json::Object &,
                                             JSONFormat::EntityIdFromJSONFn) {
  return createStringError(inconvertibleErrorCode(),
                           "intentional deserializer failure");
}

JSONFormat::AnalysisResultRegistry::Add<FailingDeserializerAnalysisResult>
    RegisterFailingDeserializerForJSON(
        serializeFailingDeserializerAnalysisResult,
        deserializeFailingDeserializerAnalysisResult);

} // namespace
