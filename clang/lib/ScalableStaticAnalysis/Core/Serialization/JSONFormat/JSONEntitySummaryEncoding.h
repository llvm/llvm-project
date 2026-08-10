//===- JSONEntitySummaryEncoding.h ------------------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_SERIALIZATION_JSONFORMAT_JSONENTITYSUMMARYENCODING_H
#define LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_SERIALIZATION_JSONFORMAT_JSONENTITYSUMMARYENCODING_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntitySummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "llvm/Support/JSON.h"

#include <map>

namespace clang::ssaf {

class JSONEntitySummaryEncoding final : public EntitySummaryEncoding {
  friend JSONFormat;

public:
  llvm::Error patch(const EntityResolutionMap &Resolution) override;

  const void *getEncodingKind() const override { return &Kind; }

  bool equals(const EntitySummaryEncoding &Other) const override;

private:
  /// Distinguishes this encoding from other formats' without RTTI. Its address
  /// is the identity; the value is irrelevant.
  static const char Kind;

  explicit JSONEntitySummaryEncoding(llvm::json::Value Data)
      : Data(std::move(Data)) {}

  llvm::Error patchEntityIdObject(llvm::json::Object &Obj,
                                  const EntityResolutionMap &Resolution,
                                  llvm::json::Value *AtVal);
  llvm::Error patchRegularObject(llvm::json::Object &Obj,
                                 const EntityResolutionMap &Resolution);
  llvm::Error patchObject(llvm::json::Object &Obj,
                          const EntityResolutionMap &Resolution);
  llvm::Error patchValue(llvm::json::Value &V,
                         const EntityResolutionMap &Resolution);

  llvm::json::Value Data;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_SERIALIZATION_JSONFORMAT_JSONENTITYSUMMARYENCODING_H
