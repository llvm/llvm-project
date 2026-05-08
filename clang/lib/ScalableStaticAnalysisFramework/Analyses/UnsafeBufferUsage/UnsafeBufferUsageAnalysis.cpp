//===- UnsafeBufferUsageAnalysis.cpp - WPA for UnsafeBufferUsage ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UnsafeBufferUsageAnalysis is a noop analysis.
//
// UnsafeBufferUsageAnalysisResult is a map from EntityIds to
// EntityPointerLevelSets
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

using namespace clang::ssaf;
using namespace llvm;

namespace {

json::Object serializeUnsafeBufferUsageAnalysisResult(
    const UnsafeBufferUsageAnalysisResult &R,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  json::Array Content;

  // Flat key-value pairs into an array of values:
  for (auto &[Id, EPLs] : R.UnsafeBuffers) {
    Content.push_back(IdToJSON(Id));
    Content.push_back(entityPointerLevelSetToJSON(EPLs, IdToJSON));
  }

  json::Object Result;

  Result[UnsafeBufferUsageAnalysisResultName] = std::move(Content);
  return Result;
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeUnsafeBufferUsageAnalysisResult(
    const json::Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content =
      Obj.getArray(UnsafeBufferUsageAnalysisResultName);

  if (!Content)
    return makeSawButExpectedError(Obj, "an object with a key %s",
                                   UnsafeBufferUsageAnalysisResultName.data());

  if (Content->size() % 2 != 0)
    return makeSawButExpectedError(*Content,
                                   "an even number of elements, got %lu",
                                   (unsigned long)Content->size());

  std::map<EntityId, EntityPointerLevelSet> UnsafeBuffers;

  for (size_t I = 0; I < Content->size(); I += 2) {
    const json::Object *IdData = (*Content)[I].getAsObject();

    if (!IdData)
      return makeSawButExpectedError((*Content)[I],
                                     "an object representing EntityId");

    auto Id = IdFromJSON(*IdData);

    if (!Id)
      return Id.takeError();

    const json::Array *EPLsData = (*Content)[I + 1].getAsArray();

    if (!EPLsData)
      return makeSawButExpectedError(
          (*Content)[I + 1], "an array representing EntityPointerLevelSet");

    auto EPLs = entityPointerLevelSetFromJSON(*EPLsData, IdFromJSON);

    if (!EPLs)
      return EPLs.takeError();
    UnsafeBuffers[*Id] = std::move(*EPLs);
  }

  auto Ret = std::make_unique<UnsafeBufferUsageAnalysisResult>();

  Ret->UnsafeBuffers = std::move(UnsafeBuffers);
  return Ret;
}

JSONFormat::AnalysisResultRegistry::Add<UnsafeBufferUsageAnalysisResult>
    RegisterUnsafeBufferUsageResultForJSON(
        serializeUnsafeBufferUsageAnalysisResult,
        deserializeUnsafeBufferUsageAnalysisResult);

class UnsafeBufferUsageAnalysis final
    : public SummaryAnalysis<UnsafeBufferUsageAnalysisResult,
                             UnsafeBufferUsageEntitySummary> {
public:
  llvm::Error add(EntityId Id,
                  const UnsafeBufferUsageEntitySummary &Summary) override {
    auto UnsafeBuffersOfEntity = getUnsafeBuffers(Summary);

    getResult().UnsafeBuffers[Id] = EntityPointerLevelSet(
        UnsafeBuffersOfEntity.begin(), UnsafeBuffersOfEntity.end());
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<UnsafeBufferUsageAnalysis>
    RegisterUnsafeBufferUsageAnalysis(
        "Whole-program unsafe buffer usage analysis");

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageAnalysisAnchorSource = 0;
