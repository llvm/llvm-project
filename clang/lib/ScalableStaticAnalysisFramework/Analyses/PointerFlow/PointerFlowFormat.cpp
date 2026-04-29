//===- PointerFlowFormat.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

using namespace clang;
using namespace ssaf;
using Object = llvm::json::Object;
using Array = llvm::json::Array;
using Value = llvm::json::Value;

ssaf::PointerFlowEntitySummary
ssaf::buildPointerFlowEntitySummary(EdgeSet Edges);

llvm::iterator_range<EdgeSet::const_iterator>
ssaf::getEdges(const PointerFlowEntitySummary &Sum);

namespace {
constexpr const char *const PointerFlowKey = "PointerFlow";
} // namespace

// Writes the 'Edges' map as an array of array of EntityPointerLevels:
// Array [
//    Array [ [src-node], [dest-node], [dest-node], ...]
//    Array [ [src-node], [dest-node], [dest-node], ...]
//    ...
// ]
static llvm::json::Object
summaryToJSON(const EntitySummary &ES,
              JSONFormat::EntityIdToJSONFn EntityId2JSON) {
  Array EdgesData;

  for (const auto &Entry :
       getEdges(static_cast<const PointerFlowEntitySummary &>(ES))) {
    Array EdgesEntryData;
    EntityPointerLevel LHS = Entry.first;

    EdgesEntryData.push_back(entityPointerLevelToJSON(LHS, EntityId2JSON));
    // Add to nodes:
    for (const auto &RHS : Entry.second)
      EdgesEntryData.push_back(entityPointerLevelToJSON(RHS, EntityId2JSON));
    EdgesData.push_back(Value(std::move(EdgesEntryData)));
  }
  return Object{{PointerFlowKey, Value(std::move(EdgesData))}};
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
summaryFromJSON(const Object &Data, EntityIdTable &,
                JSONFormat::EntityIdFromJSONFn EntityIdFromJSON) {
  const Value *EdgesData = Data.get(PointerFlowKey);

  if (!EdgesData)
    return makeSawButExpectedError(
        Object(Data), "a JSON object with the key: %s", PointerFlowKey);

  EdgeSet Edges;
  const auto *EdgesDataAsArr = EdgesData->getAsArray();

  if (!EdgesDataAsArr)
    return makeSawButExpectedError(
        *EdgesData, "a JSON array of array of EntityPointerLevels");
  for (const auto &EdgesEntryData : *EdgesDataAsArr) {
    const auto *EPLArray = EdgesEntryData.getAsArray();

    if (!EPLArray || EPLArray->size() <= 1)
      return makeSawButExpectedError(
          EdgesEntryData, "a JSON array of EntityPointerLevels with a size "
                          "greater than 1: [lhs, rhs, rhs, ...]");

    auto SrcEPL = entityPointerLevelFromJSON((*EPLArray)[0], EntityIdFromJSON);

    if (!SrcEPL)
      return SrcEPL.takeError();
    for (const auto &EPLData : llvm::drop_begin(*EPLArray)) {
      auto EPL = entityPointerLevelFromJSON(EPLData, EntityIdFromJSON);
      if (!EPL)
        return EPL.takeError();
      Edges[*SrcEPL].insert(*EPL);
    }
  }
  return std::make_unique<PointerFlowEntitySummary>(
      buildPointerFlowEntitySummary(std::move(Edges)));
}

namespace {
struct PointerFlowJSONFormatInfo final : JSONFormat::FormatInfo {
  PointerFlowJSONFormatInfo()
      : JSONFormat::FormatInfo(PointerFlowEntitySummary::summaryName(),
                               summaryToJSON, summaryFromJSON) {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<PointerFlowJSONFormatInfo>
    RegisterPointerFlowJSONFormatInfo(
        "PointerFlow", "JSON Format info for PointerFlowEntitySummary");

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int PointerFlowSSAFJSONFormatAnchorSource = 0;
