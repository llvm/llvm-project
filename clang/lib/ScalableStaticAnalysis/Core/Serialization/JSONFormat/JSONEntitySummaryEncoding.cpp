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

const char JSONEntitySummaryEncoding::Kind = 0;

bool JSONEntitySummaryEncoding::equals(
    const EntitySummaryEncoding &Other) const {
  if (Other.getEncodingKind() != getEncodingKind()) {
    return false;
  }
  // json::Value's operator== compares objects as maps, so key order does not
  // affect the result.
  return Data == static_cast<const JSONEntitySummaryEncoding &>(Other).Data;
}

llvm::Error JSONEntitySummaryEncoding::patchEntityIdObject(
    llvm::json::Object &Obj, const EntityResolutionMap &Resolution,
    llvm::json::Value *AtVal) {

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
  auto It = Resolution.find(OldId);
  if (It == Resolution.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToPatchEntityIdNotInTable,
                                OldId)
        .build();
  }

  *AtVal = static_cast<uint64_t>(JSONFormat::getIndex(It->second));

  return llvm::Error::success();
}

llvm::Error JSONEntitySummaryEncoding::patchRegularObject(
    llvm::json::Object &Obj, const EntityResolutionMap &Resolution) {
  for (auto &[Key, Val] : Obj) {
    if (auto Err = patchValue(Val, Resolution)) {
      return Err;
    }
  }
  return llvm::Error::success();
}

llvm::Error
JSONEntitySummaryEncoding::patchObject(llvm::json::Object &Obj,
                                       const EntityResolutionMap &Resolution) {

  llvm::json::Value *AtVal = Obj.get(JSONEntityIdKey);
  return AtVal ? patchEntityIdObject(Obj, Resolution, AtVal)
               : patchRegularObject(Obj, Resolution);
}

llvm::Error
JSONEntitySummaryEncoding::patchValue(llvm::json::Value &V,
                                      const EntityResolutionMap &Resolution) {
  if (llvm::json::Object *Obj = V.getAsObject()) {
    if (auto Err = patchObject(*Obj, Resolution)) {
      return Err;
    }
  } else if (llvm::json::Array *Arr = V.getAsArray()) {
    for (auto &Val : *Arr) {
      if (auto Err = patchValue(Val, Resolution)) {
        return Err;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
JSONEntitySummaryEncoding::patch(const EntityResolutionMap &Resolution) {
  return patchValue(Data, Resolution);
}

} // namespace clang::ssaf
