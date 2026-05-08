//===- UnsafeBufferUsageFormat.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageTest.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <cstdint>

using namespace clang;
using namespace ssaf;
using Array = llvm::json::Array;
using Object = llvm::json::Object;

static constexpr llvm::StringLiteral SummarySerializationKey = "UnsafeBuffers";

extern UnsafeBufferUsageEntitySummary
ssaf::buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet UnsafeBuffers);

extern llvm::iterator_range<EntityPointerLevelSet::const_iterator>
ssaf::getUnsafeBuffers(const UnsafeBufferUsageEntitySummary &S);

static Object serialize(const EntitySummary &S,
                        JSONFormat::EntityIdToJSONFn Fn) {
  const auto &SS = static_cast<const UnsafeBufferUsageEntitySummary &>(S);
  Array UnsafeBuffersData;

  for (const auto &EPL : getUnsafeBuffers(SS))
    UnsafeBuffersData.push_back(entityPointerLevelToJSON(EPL, Fn));
  return Object{{SummarySerializationKey.data(), std::move(UnsafeBuffersData)}};
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserializeImpl(const Object &Data, JSONFormat::EntityIdFromJSONFn Fn) {
  const Array *UnsafeBuffersData =
      Data.getArray(SummarySerializationKey.data());

  if (!UnsafeBuffersData)
    return makeSawButExpectedError(Object(Data), "an Object with a key %s",
                                   SummarySerializationKey.data());

  EntityPointerLevelSet EPLs;

  for (const auto &EltData : *UnsafeBuffersData) {
    llvm::Expected<EntityPointerLevel> EPL =
        entityPointerLevelFromJSON(EltData, Fn);

    if (!EPL)
      return EPL.takeError();
    EPLs.insert(*EPL);
  }
  return std::make_unique<UnsafeBufferUsageEntitySummary>(
      buildUnsafeBufferUsageEntitySummary(std::move(EPLs)));
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserialize(const Object &Data, EntityIdTable &,
            JSONFormat::EntityIdFromJSONFn Fn) {
  return deserializeImpl(Data, Fn);
}

namespace {
struct UnsafeBufferUsageJSONFormatInfo final : JSONFormat::FormatInfo {
  UnsafeBufferUsageJSONFormatInfo()
      : JSONFormat::FormatInfo(UnsafeBufferUsageEntitySummary::summaryName(),
                               serialize, deserialize) {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    UnsafeBufferUsageJSONFormatInfo>
    RegisterUnsafeBufferUsageJSONFormatInfo(
        UnsafeBufferUsageEntitySummary::Name,
        "JSON Format info for UnsafeBufferUsageEntitySummary");

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageSSAFJSONFormatAnchorSource = 0;

// For unit test:
llvm::Expected<std::unique_ptr<EntitySummary>>
ssaf::serializeDeserializeRoundTrip(
    const UnsafeBufferUsageEntitySummary &S,
    std::function<uint64_t(EntityId)> IdToIntFn,
    std::function<llvm::Expected<EntityId>(uint64_t)> IdFromIntFn) {

  auto IdToJson = [&IdToIntFn](EntityId Id) -> Object {
    return Object({{"@", IdToIntFn(Id)}});
  };
  auto IdFromJson =
      [&IdFromIntFn](const Object &O) -> llvm::Expected<EntityId> {
    const auto *Int = O.get("@");

    if (Int && Int->getAsUINT64())
      return IdFromIntFn(*Int->getAsUINT64());
    return llvm::createStringError("failed to get EntityId from Object");
  };

  return deserializeImpl(serialize(S, IdToJson), IdFromJson);
}
