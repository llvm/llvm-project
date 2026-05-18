//===- CallGraphJSONFormat.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/CallGraph/CallGraphSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include <memory>

using namespace llvm;
using namespace clang;
using namespace ssaf;

static const char *FailedToReadObjectAtField =
    "failed to read {0} from field '{1}': expected JSON {2}";
static const char *FailedToReadObjectAtIndex =
    "failed to read {0} from index '{1}': expected JSON {2}";
static const char *ReadingFromField = "reading {0} from field '{1}'";
static const char *ReadingFromIndex = "reading {0} from index '{1}'";

static json::Object serialize(const EntitySummary &Summary,
                              JSONFormat::EntityIdToJSONFn ToJSON) {
  const auto &S = static_cast<const CallGraphSummary &>(Summary);

  json::Array DirectCalleesArray;
  DirectCalleesArray.reserve(S.DirectCallees.size());
  append_range(DirectCalleesArray, map_range(S.DirectCallees, ToJSON));

  json::Array VirtualCalleesArray;
  VirtualCalleesArray.reserve(S.VirtualCallees.size());
  append_range(VirtualCalleesArray, map_range(S.VirtualCallees, ToJSON));

  return json::Object{
      {"pretty_name", json::Value(S.PrettyName)},
      {"direct_callees", std::move(DirectCalleesArray)},
      {"virtual_callees", std::move(VirtualCalleesArray)},
      {"def",
       json::Object{
           {"file", json::Value(S.Definition.File)},
           {"line", json::Value(S.Definition.Line)},
           {"col", json::Value(S.Definition.Column)},
       }},
  };
}

static Expected<std::unique_ptr<EntitySummary>>
deserialize(const json::Object &Obj, EntityIdTable &IdTable,
            JSONFormat::EntityIdFromJSONFn FromJSON) {
  auto Result = std::make_unique<CallGraphSummary>();

  auto PrettyName = Obj.getString("pretty_name");
  if (!PrettyName) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "PrettyName",
                                "pretty_name", "string")
        .build();
  }
  Result->PrettyName = PrettyName->str();

  const json::Array *CalleesArray = Obj.getArray("direct_callees");
  if (!CalleesArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "DirectCallees",
                                "direct_callees", "array")
        .build();
  }
  for (const auto &[Index, Value] : llvm::enumerate(*CalleesArray)) {
    const json::Object *CalleeObj = Value.getAsObject();
    if (!CalleeObj) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  FailedToReadObjectAtIndex, "EntityId", Index,
                                  "object")
          .context(ReadingFromField, "DirectCallees", "direct_callees")
          .build();
    }
    auto ExpectedId = FromJSON(*CalleeObj);
    if (!ExpectedId) {
      return ErrorBuilder::wrap(ExpectedId.takeError())
          .context(ReadingFromIndex, "EntityId", Index)
          .context(ReadingFromField, "DirectCallees", "direct_callees")
          .build();
    }
    Result->DirectCallees.insert(*ExpectedId);
  }

  const json::Array *VirtualCalleesArray = Obj.getArray("virtual_callees");
  if (!VirtualCalleesArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "VirtualCallees",
                                "virtual_callees", "array")
        .build();
  }
  for (const auto &[Index, Value] : llvm::enumerate(*VirtualCalleesArray)) {
    const json::Object *CalleeObj = Value.getAsObject();
    if (!CalleeObj) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  FailedToReadObjectAtIndex, "EntityId", Index,
                                  "object")
          .context(ReadingFromField, "VirtualCallees", "virtual_callees")
          .build();
    }
    auto ExpectedId = FromJSON(*CalleeObj);
    if (!ExpectedId) {
      return ErrorBuilder::wrap(ExpectedId.takeError())
          .context(ReadingFromIndex, "EntityId", Index)
          .context(ReadingFromField, "VirtualCallees", "virtual_callees")
          .build();
    }
    Result->VirtualCallees.insert(*ExpectedId);
  }

  const json::Object *DefObj = Obj.getObject("def");
  if (!DefObj) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "SourceLocation",
                                "def", "object")
        .build();
  }
  auto File = DefObj->getString("file");
  if (!File) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "File", "file",
                                "string")
        .context(ReadingFromField, "SourceLocation", "def")
        .build();
  }
  auto Line = DefObj->getInteger("line");
  if (!Line) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "Line", "line",
                                "number")
        .context(ReadingFromField, "SourceLocation", "def")
        .build();
  }
  auto Col = DefObj->getInteger("col");
  if (!Col) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                FailedToReadObjectAtField, "Column", "col",
                                "number")
        .context(ReadingFromField, "SourceLocation", "def")
        .build();
  }
  Result->Definition = {
      File->str(),
      static_cast<unsigned>(*Line),
      static_cast<unsigned>(*Col),
  };

  return std::move(Result);
}

namespace {
struct CallGraphJSONFormatInfo final : JSONFormat::FormatInfo {
  CallGraphJSONFormatInfo()
      : JSONFormat::FormatInfo(SummaryName(CallGraphSummary::Name.str()),
                               serialize, deserialize) {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<CallGraphJSONFormatInfo>
    RegisterFormatInfo(CallGraphSummary::Name,
                       "JSON Format info for CallGraph summary");

// This anchor is used to force the linker to link in the generated object file
// and thus register the JSON format for CallGraphSummary.
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int CallGraphJSONFormatAnchorSource = 0;
