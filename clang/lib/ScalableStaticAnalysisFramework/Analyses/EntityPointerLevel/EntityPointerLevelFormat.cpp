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
