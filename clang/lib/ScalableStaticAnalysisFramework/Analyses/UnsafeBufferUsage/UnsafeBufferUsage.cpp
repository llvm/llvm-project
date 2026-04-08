//===- UnsafeBufferUsage.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageTest.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <cstdint>

using namespace clang;
using namespace ssaf;
using Array = llvm::json::Array;
using Object = llvm::json::Object;

static constexpr llvm::StringLiteral SummarySerializationKey = "UnsafeBuffers";

EntityPointerLevel ssaf::buildEntityPointerLevel(EntityId Id, unsigned PtrLv) {
  return EntityPointerLevel(Id, PtrLv);
}

UnsafeBufferUsageEntitySummary
ssaf::buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet UnsafeBuffers) {
  return UnsafeBufferUsageEntitySummary(std::move(UnsafeBuffers));
}

llvm::iterator_range<EntityPointerLevelSet::const_iterator>
ssaf::getUnsafeBuffers(const UnsafeBufferUsageEntitySummary &S) {
  return llvm::make_range(S.UnsafeBuffers.begin(), S.UnsafeBuffers.end());
}

static Object serialize(const EntitySummary &S,
                        JSONFormat::EntityIdToJSONFn Fn) {
  const auto &SS = static_cast<const UnsafeBufferUsageEntitySummary &>(S);
  Array UnsafeBuffersData;

  for (const auto &EPL : getUnsafeBuffers(SS))
    UnsafeBuffersData.push_back(
        Array{Fn(EPL.getEntity()), EPL.getPointerLevel()});
  return Object{{SummarySerializationKey.data(), std::move(UnsafeBuffersData)}};
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserializeImpl(const Object &Data, JSONFormat::EntityIdFromJSONFn Fn) {
  const Array *UnsafeBuffersData =
      Data.getArray(SummarySerializationKey.data());

  if (!UnsafeBuffersData)
    return llvm::createStringError("expected a json::Object with a key %s",
                                   SummarySerializationKey.data());

  EntityPointerLevelSet EPLs;

  for (const auto &EltData : *UnsafeBuffersData) {
    const Array *EltDataAsArr = EltData.getAsArray();

    if (!EltDataAsArr || EltDataAsArr->size() != 2)
      return llvm::createStringError("expected a json::Array of size 2");

    const Object *IdData = (*EltDataAsArr)[0].getAsObject();
    std::optional<uint64_t> PtrLvData = (*EltDataAsArr)[1].getAsInteger();

    if (!IdData || !PtrLvData)
      return llvm::createStringError("expected a json::Value of integer type");

    llvm::Expected<EntityId> Id = Fn(*IdData);

    if (!Id)
      return Id.takeError();
    EPLs.insert(buildEntityPointerLevel(Id.get(), *PtrLvData));
  }
  return std::make_unique<UnsafeBufferUsageEntitySummary>(
      buildUnsafeBufferUsageEntitySummary(std::move(EPLs)));
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserialize(const Object &Data, EntityIdTable &,
            JSONFormat::EntityIdFromJSONFn Fn) {
  return deserializeImpl(Data, Fn);
}

struct UnsafeBufferUsageJSONFormatInfo : JSONFormat::FormatInfo {
  UnsafeBufferUsageJSONFormatInfo()
      : JSONFormat::FormatInfo(UnsafeBufferUsageEntitySummary::summaryName(),
                               serialize, deserialize) {}
};

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
