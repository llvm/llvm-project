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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include <memory>

using namespace llvm;
using namespace clang;
using namespace ssaf;

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
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'pretty_name'");
  }
  Result->PrettyName = PrettyName->str();

  const json::Array *CalleesArray = Obj.getArray("direct_callees");
  if (!CalleesArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'direct_callees'");
  }
  for (const auto &[Index, Value] : llvm::enumerate(*CalleesArray)) {
    const json::Object *CalleeObj = Value.getAsObject();
    if (!CalleeObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "direct_callees element at index %zu is not a JSON object", Index);
    }
    auto ExpectedId = FromJSON(*CalleeObj);
    if (!ExpectedId) {
      return createStringError(
          inconvertibleErrorCode(),
          "invalid entity id in direct_callees at index %zu: %s", Index,
          toString(ExpectedId.takeError()).c_str());
    }
    Result->DirectCallees.insert(*ExpectedId);
  }

  const json::Array *VirtualCalleesArray = Obj.getArray("virtual_callees");
  if (!VirtualCalleesArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'virtual_callees'");
  }
  for (const auto &[Index, Value] : llvm::enumerate(*VirtualCalleesArray)) {
    const json::Object *CalleeObj = Value.getAsObject();
    if (!CalleeObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "virtual_callees element at index %zu is not a JSON object", Index);
    }
    auto ExpectedId = FromJSON(*CalleeObj);
    if (!ExpectedId) {
      return createStringError(
          inconvertibleErrorCode(),
          "invalid entity id in virtual_callees at index %zu: %s", Index,
          toString(ExpectedId.takeError()).c_str());
    }
    Result->VirtualCallees.insert(*ExpectedId);
  }

  const json::Object *DefObj = Obj.getObject("def");
  if (!DefObj) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'def'");
  }
  auto File = DefObj->getString("file");
  if (!File) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'def.file'");
  }
  auto Line = DefObj->getInteger("line");
  if (!Line) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'def.line'");
  }
  auto Col = DefObj->getInteger("col");
  if (!Col) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'def.col'");
  }
  Result->Definition = {File->str(), static_cast<unsigned>(*Line),
                        static_cast<unsigned>(*Col)};

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
