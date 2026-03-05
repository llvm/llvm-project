//===- JSONEntitySummaryEncoding.h ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Opaque JSON-based entity summary encoding used by JSONFormat. Stores raw
// EntitySummary JSON blobs and patches embedded entity ID references without
// requiring knowledge of the analysis schema.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_JSONENTITYSUMMARYENCODING_H
#define CLANG_LIB_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_JSONENTITYSUMMARYENCODING_H

#include "clang/Analysis/Scalable/EntityLinker/EntitySummaryEncoding.h"
#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "llvm/Support/JSON.h"

#include <map>

namespace clang::ssaf {

class JSONEntitySummaryEncoding final : public EntitySummaryEncoding {
  friend JSONFormat;

public:
  llvm::Error
  patch(const std::map<EntityId, EntityId> &EntityResolutionTable) override;

private:
  explicit JSONEntitySummaryEncoding(llvm::json::Value Data)
      : Data(std::move(Data)) {}

  llvm::Expected<bool>
  patchEntityIdObject(llvm::json::Object &Obj,
                      const std::map<EntityId, EntityId> &Table);
  llvm::Error patchObject(llvm::json::Object &Obj,
                          const std::map<EntityId, EntityId> &Table);
  llvm::Error patchValue(llvm::json::Value &V,
                         const std::map<EntityId, EntityId> &Table);

  llvm::json::Value Data;
};

} // namespace clang::ssaf

#endif // CLANG_LIB_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_JSONENTITYSUMMARYENCODING_H
