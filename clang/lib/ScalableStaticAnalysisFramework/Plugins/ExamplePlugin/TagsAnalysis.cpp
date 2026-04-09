//===- TagsAnalysis.cpp - Tags analysis for ExamplePlugin -----------------===//
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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;
using example_plugin::TagsAnalysisResult;

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
  TagsArray.reserve(S.Tags.size());
  llvm::append_range(TagsArray, S.Tags);
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
// TagsAnalysisResult serialization
//
// Sorted, deduplicated list of all tags seen across entities. Serialized as:
//   { "tags": ["tag1", "tag2", ...] }
//===----------------------------------------------------------------------===//

json::Object serializeTagsAnalysisResult(const TagsAnalysisResult &R,
                                         JSONFormat::EntityIdToJSONFn) {
  json::Array TagsArray;
  TagsArray.reserve(R.Tags.size());
  llvm::append_range(TagsArray, R.Tags);
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

} // namespace
