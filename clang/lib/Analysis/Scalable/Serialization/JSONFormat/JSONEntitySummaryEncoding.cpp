//===- JSONEntitySummaryEncoding.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONEntitySummaryEncoding.h"
#include "JSONFormatImpl.h"

namespace clang::ssaf {

llvm::Expected<bool> JSONEntitySummaryEncoding::patchEntityIdObject(
    llvm::json::Object &Obj, const std::map<EntityId, EntityId> &Table) {

  llvm::json::Value *AtVal = Obj.get(JSONEntityIdKey);
  if (!AtVal) {
    return false;
  }

  if (Obj.size() != 1) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadEntityIdObject,
                                JSONEntityIdKey)
        .build();
  }

  std::optional<uint64_t> OptEntityIdIndex = AtVal->getAsUINT64();
  if (!OptEntityIdIndex) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadEntityIdObject,
                                JSONEntityIdKey)
        .build();
  }

  auto OldId = JSONFormat::makeEntityId(*OptEntityIdIndex);
  auto It = Table.find(OldId);
  if (It == Table.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToPatchEntityIdNotInTable,
                                OldId)
        .build();
  }

  *AtVal = static_cast<uint64_t>(JSONFormat::getIndex(It->second));

  return true;
}

llvm::Error JSONEntitySummaryEncoding::patchObject(
    llvm::json::Object &Obj, const std::map<EntityId, EntityId> &Table) {

  auto ExpectedIsEntityId = patchEntityIdObject(Obj, Table);

  if (!ExpectedIsEntityId) {
    return ExpectedIsEntityId.takeError();
  }

  bool IsEntityId = *ExpectedIsEntityId;

  if (!IsEntityId) {
    for (auto &[Key, Val] : Obj) {
      if (auto Err = patchValue(Val, Table)) {
        return Err;
      }
    }
  }

  return llvm::Error::success();
}

llvm::Error JSONEntitySummaryEncoding::patchValue(
    llvm::json::Value &V, const std::map<EntityId, EntityId> &Table) {
  if (llvm::json::Object *Obj = V.getAsObject()) {
    if (auto Err = patchObject(*Obj, Table)) {
      return Err;
    }
  } else if (llvm::json::Array *Arr = V.getAsArray()) {
    for (auto &Val : *Arr) {
      if (auto Err = patchValue(Val, Table)) {
        return Err;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error JSONEntitySummaryEncoding::patch(
    const std::map<EntityId, EntityId> &EntityResolutionTable) {
  return patchValue(Data, EntityResolutionTable);
}

} // namespace clang::ssaf
