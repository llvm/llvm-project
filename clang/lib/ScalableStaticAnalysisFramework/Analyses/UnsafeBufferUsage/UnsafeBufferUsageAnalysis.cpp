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
  json::Object Result;

  Result[UnsafeBufferUsageAnalysisResultName] =
      entityPointerLevelMapToJSON(R.UnsafeBuffers, IdToJSON);
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

  auto UnsafeBuffers = entityPointerLevelMapFromJSON(*Content, IdFromJSON);

  if (!UnsafeBuffers)
    return UnsafeBuffers.takeError();

  auto Ret = std::make_unique<UnsafeBufferUsageAnalysisResult>();

  Ret->UnsafeBuffers = std::move(*UnsafeBuffers);
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
