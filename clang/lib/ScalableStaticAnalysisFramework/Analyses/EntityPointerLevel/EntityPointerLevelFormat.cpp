//===- EntityPointerLevelFormat.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

using namespace clang;
using namespace ssaf;

// Writes an EntityPointerLevel as
// Array [
//   Object { "@" : [entity-id]},
//   [pointer-level-integer]
// ]
llvm::json::Value clang::ssaf::entityPointerLevelToJSON(
    const EntityPointerLevel &EPL, JSONFormat::EntityIdToJSONFn EntityId2JSON) {
  return llvm::json::Array{EntityId2JSON(EPL.getEntity()),
                           llvm::json::Value(EPL.getPointerLevel())};
}

llvm::Expected<EntityPointerLevel> clang::ssaf::entityPointerLevelFromJSON(
    const llvm::json::Value &EPLData,
    JSONFormat::EntityIdFromJSONFn EntityIdFromJSON) {
  auto *AsArr = EPLData.getAsArray();

  if (!AsArr || AsArr->size() != 2)
    return makeSawButExpectedError(
        EPLData, "an array with exactly two elements representing "
                 "EntityId and PointerLevel, respectively");

  auto *EntityIdObj = (*AsArr)[0].getAsObject();

  if (!EntityIdObj)
    return makeSawButExpectedError((*AsArr)[0],
                                   "an object representing EntityId");

  llvm::Expected<EntityId> Id = EntityIdFromJSON(*EntityIdObj);

  if (!Id)
    return Id.takeError();

  std::optional<uint64_t> PtrLv = (*AsArr)[1].getAsInteger();

  if (!PtrLv)
    return makeSawButExpectedError((*AsArr)[1],
                                   "an integer representing PointerLevel");

  return buildEntityPointerLevel(*Id, *PtrLv);
}

llvm::json::Array clang::ssaf::entityPointerLevelSetToJSON(
    llvm::iterator_range<EntityPointerLevelSet::const_iterator> EPLs,
    JSONFormat::EntityIdToJSONFn EntityId2JSON) {
  llvm::json::Array Result;

  for (const auto &EPL : EPLs)
    Result.push_back(entityPointerLevelToJSON(EPL, EntityId2JSON));
  return Result;
}

Expected<EntityPointerLevelSet> clang::ssaf::entityPointerLevelSetFromJSON(
    const llvm::json::Array &EPLsData,
    JSONFormat::EntityIdFromJSONFn EntityIdFromJSON) {
  EntityPointerLevelSet EPLs;

  for (const auto &EltData : EPLsData) {
    llvm::Expected<EntityPointerLevel> EPL =
        entityPointerLevelFromJSON(EltData, EntityIdFromJSON);

    if (!EPL)
      return EPL.takeError();
    EPLs.insert(*EPL);
  }
  return EPLs;
}

llvm::json::Array clang::ssaf::entityPointerLevelMapToJSON(
    const std::map<EntityId, EntityPointerLevelSet> &Map,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  llvm::json::Array Content;

  for (const auto &[Id, EPLs] : Map) {
    Content.push_back(IdToJSON(Id));
    Content.push_back(entityPointerLevelSetToJSON(EPLs, IdToJSON));
  }
  return Content;
}

Expected<std::map<EntityId, EntityPointerLevelSet>>
clang::ssaf::entityPointerLevelMapFromJSON(
    const llvm::json::Array &Content,
    JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  if (Content.size() % 2 != 0)
    return makeSawButExpectedError(Content,
                                   "an even number of elements, got %lu",
                                   static_cast<size_t>(Content.size()));

  std::map<EntityId, EntityPointerLevelSet> Result;

  for (size_t I = 0; I < Content.size(); I += 2) {
    const llvm::json::Object *IdData = Content[I].getAsObject();

    if (!IdData)
      return makeSawButExpectedError(Content[I],
                                     "an object representing EntityId");

    auto Id = IdFromJSON(*IdData);

    if (!Id)
      return Id.takeError();

    const llvm::json::Array *EPLsData = Content[I + 1].getAsArray();

    if (!EPLsData)
      return makeSawButExpectedError(
          Content[I + 1], "an array representing EntityPointerLevelSet");

    auto EPLs = entityPointerLevelSetFromJSON(*EPLsData, IdFromJSON);

    if (!EPLs)
      return EPLs.takeError();
    Result[*Id] = std::move(*EPLs);
  }
  return Result;
}
