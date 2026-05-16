//===- PointerFlowAnalysis.cpp - WPA for PointerFlow ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlowFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

using namespace clang::ssaf;
using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// PointerFlowAnalysis---converts PointerFlowEntitySummary(s) in an LUSummary to
// a PointerFlowAnalysisResult
//===----------------------------------------------------------------------===//

// Serialized as a flat array of alternating [EntityId, EdgesArray, ...] pairs.
json::Object
serializePointerFlowAnalysisResult(const PointerFlowAnalysisResult &R,
                                   JSONFormat::EntityIdToJSONFn IdToJSON) {
  json::Array Content;

  for (const auto &[Id, EntityEdges] : R.Edges) {
    Content.push_back(IdToJSON(Id));
    Content.push_back(json::Value(edgeSetToJSON(EntityEdges, IdToJSON)));
  }

  json::Object Result;

  Result[PointerFlowAnalysisResultName] = std::move(Content);
  return Result;
}

Expected<std::unique_ptr<AnalysisResult>> deserializePointerFlowAnalysisResult(
    const json::Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content = Obj.getArray(PointerFlowAnalysisResultName);

  if (!Content)
    return makeSawButExpectedError(Obj, "an object with a key %s",
                                   PointerFlowAnalysisResultName.data());

  if (Content->size() % 2 != 0)
    return makeSawButExpectedError(*Content,
                                   "an even number of elements, got %lu",
                                   static_cast<size_t>(Content->size()));

  std::map<EntityId, EdgeSet> Edges;

  for (size_t I = 0; I < Content->size(); I += 2) {
    const json::Object *IdData = (*Content)[I].getAsObject();

    if (!IdData)
      return makeSawButExpectedError((*Content)[I],
                                     "an object representing EntityId");

    auto Id = IdFromJSON(*IdData);

    if (!Id)
      return Id.takeError();

    const json::Array *EdgesData = (*Content)[I + 1].getAsArray();

    if (!EdgesData)
      return makeSawButExpectedError((*Content)[I + 1],
                                     "an array of arrays representing EdgeSet");

    auto EntityEdges = edgeSetFromJSON(*EdgesData, IdFromJSON);

    if (!EntityEdges)
      return EntityEdges.takeError();
    Edges[*Id] = std::move(*EntityEdges);
  }

  auto Ret = std::make_unique<PointerFlowAnalysisResult>();

  Ret->Edges = std::move(Edges);
  return Ret;
}

JSONFormat::AnalysisResultRegistry::Add<PointerFlowAnalysisResult>
    RegisterPointerFlowResultForJSON(serializePointerFlowAnalysisResult,
                                     deserializePointerFlowAnalysisResult);

class PointerFlowAnalysis final
    : public SummaryAnalysis<PointerFlowAnalysisResult,
                             PointerFlowEntitySummary> {
public:
  llvm::Error add(EntityId Id,
                  const PointerFlowEntitySummary &Summary) override {
    auto EdgesOfEntity = getEdges(Summary);

    getResult().Edges[Id] = EdgeSet(EdgesOfEntity.begin(), EdgesOfEntity.end());
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<PointerFlowAnalysis>
    RegisterPointerFlowAnalysis("Whole-program pointer flow analysis");

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int PointerFlowAnalysisAnchorSource = 0;
