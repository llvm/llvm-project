//===- VirtualMethodFamilyFormat.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;

using llvm::Expected;
using Object = llvm::json::Object;
using Array = llvm::json::Array;

namespace {
constexpr llvm::StringLiteral KeyParamEntities = "param_entities";
constexpr llvm::StringLiteral KeyReturnEntity = "return_entity";
constexpr llvm::StringLiteral KeyOverriddenMethods = "overridden_methods";

constexpr llvm::StringLiteral KeyRetAndParamFamilyIds = "families";
constexpr llvm::StringLiteral KeyParamId = "pid";
constexpr llvm::StringLiteral KeyFamilyId = "fid";
constexpr llvm::StringLiteral KeyOwnerMethodId = "oid";
} // namespace

static Array entityIdVectorToJSON(const std::vector<EntityId> &Ids,
                                  JSONFormat::EntityIdToJSONFn IdToJSON) {
  Array Result;
  Result.reserve(Ids.size());
  for (EntityId Id : Ids)
    Result.push_back(IdToJSON(Id));
  return Result;
}

static Expected<std::vector<EntityId>>
entityIdVectorFromJSON(const Array &Arr,
                       JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  std::vector<EntityId> Result;
  Result.reserve(Arr.size());
  for (const auto &V : Arr) {
    const Object *Obj = V.getAsObject();
    if (!Obj)
      return makeSawButExpectedError(V, "an object representing EntityId");
    auto Id = IdFromJSON(*Obj);
    if (!Id)
      return Id.takeError();
    Result.push_back(*Id);
  }
  return Result;
}

//===----------------------------------------------------------------------===//
// VirtualMethodSummary <-> JSON
//===----------------------------------------------------------------------===//

static Object
serializeVirtualMethodSummary(const EntitySummary &ES,
                              JSONFormat::EntityIdToJSONFn IdToJSON) {
  const auto &S = static_cast<const VirtualMethodSummary &>(ES);
  Object Out;
  Out[KeyParamEntities] = entityIdVectorToJSON(S.ParamEntities, IdToJSON);
  if (S.ReturnEntity.has_value())
    Out[KeyReturnEntity] = IdToJSON(*S.ReturnEntity);
  Out[KeyOverriddenMethods] =
      entityIdVectorToJSON(S.OverriddenMethods, IdToJSON);
  return Out;
}

static Expected<std::unique_ptr<EntitySummary>>
deserializeVirtualMethodSummary(const Object &Obj, EntityIdTable &,
                                JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  auto Result = std::make_unique<VirtualMethodSummary>();

  const Array *ParamArr = Obj.getArray(KeyParamEntities);
  if (!ParamArr)
    return makeSawButExpectedError(Obj, "an object with an array field '%s'",
                                   KeyParamEntities.data());
  auto Params = entityIdVectorFromJSON(*ParamArr, IdFromJSON);
  if (!Params)
    return Params.takeError();
  Result->ParamEntities = std::move(*Params);

  if (const Object *RE = Obj.getObject(KeyReturnEntity)) {
    auto Id = IdFromJSON(*RE);
    if (!Id)
      return Id.takeError();
    Result->ReturnEntity = *Id;
  }

  // Tolerant: absent key means no override edges (a root virtual method).
  if (const Array *OMArr = Obj.getArray(KeyOverriddenMethods)) {
    auto OM = entityIdVectorFromJSON(*OMArr, IdFromJSON);
    if (!OM)
      return OM.takeError();
    Result->OverriddenMethods = std::move(*OM);
  }

  return std::move(Result);
}

//===----------------------------------------------------------------------===//
// VirtualMethodFamilyAnalysisResult <-> JSON
//===----------------------------------------------------------------------===//

static Object serializeVirtualMethodFamilyAnalysisResult(
    const VirtualMethodFamilyAnalysisResult &R,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  Array FamilyDataArr;
  for (const auto &[ParamId, Data] : R.RetAndParamData) {
    Object Item;
    Item[KeyParamId] = IdToJSON(ParamId);
    Item[KeyFamilyId] = IdToJSON(Data.FamilyId);
    Item[KeyOwnerMethodId] = IdToJSON(Data.OwnerMethodId);
    FamilyDataArr.push_back(std::move(Item));
  }

  Object Out;
  Out[KeyRetAndParamFamilyIds] = std::move(FamilyDataArr);
  return Out;
}

static Expected<std::unique_ptr<AnalysisResult>>
deserializeVirtualMethodFamilyAnalysisResult(
    const Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const Array *FamilyDataArr = Obj.getArray(KeyRetAndParamFamilyIds);
  if (!FamilyDataArr)
    return makeSawButExpectedError(Obj, "an object with an array field '%s'",
                                   KeyRetAndParamFamilyIds.data());

  auto Result = std::make_unique<VirtualMethodFamilyAnalysisResult>();

  for (const auto &V : *FamilyDataArr) {
    const Object *Item = V.getAsObject();
    if (!Item)
      return makeSawButExpectedError(V, "an object {pid, fid, oid}");
    const Object *ParamObj = Item->getObject(KeyParamId);
    const Object *FamilyObj = Item->getObject(KeyFamilyId);
    const Object *OwnerMethodObj = Item->getObject(KeyOwnerMethodId);
    if (!ParamObj || !FamilyObj || !OwnerMethodObj)
      return makeSawButExpectedError(
          *Item, "an object with fields {'%s', '%s', '%s'}", KeyParamId.data(),
          KeyFamilyId.data(), KeyOwnerMethodId.data());
    auto ParamId = IdFromJSON(*ParamObj);
    if (!ParamId)
      return ParamId.takeError();
    auto FamilyId = IdFromJSON(*FamilyObj);
    if (!FamilyId)
      return FamilyId.takeError();
    auto OwnerMethodId = IdFromJSON(*OwnerMethodObj);
    if (!OwnerMethodId)
      return OwnerMethodId.takeError();
    Result->RetAndParamData.insert({*ParamId, {*FamilyId, *OwnerMethodId}});
  }
  return std::move(Result);
}

namespace {

struct VirtualMethodSummaryJSONFormatInfo final : JSONFormat::FormatInfo {
  VirtualMethodSummaryJSONFormatInfo()
      : JSONFormat::FormatInfo(VirtualMethodSummary::summaryName(),
                               serializeVirtualMethodSummary,
                               deserializeVirtualMethodSummary) {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    VirtualMethodSummaryJSONFormatInfo>
    RegisterJSONFormat(VirtualMethodSummary::Name,
                       "JSON Format info for VirtualMethodSummary");

static JSONFormat::AnalysisResultRegistry::Add<
    VirtualMethodFamilyAnalysisResult>
    RegisterResultJSONFormat(serializeVirtualMethodFamilyAnalysisResult,
                             deserializeVirtualMethodFamilyAnalysisResult);

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int VirtualMethodFamilyJSONFormatAnchorSource = 0;
} // namespace clang::ssaf
