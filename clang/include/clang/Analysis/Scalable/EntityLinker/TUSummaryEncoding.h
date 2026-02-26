//===- TUSummaryEncoding.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TUSummaryEncoding class, which represents a
// translation unit summary in its serialized, format-specific encoding.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_TUSUMMARYENCODING_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_TUSUMMARYENCODING_H

#include "clang/Analysis/Scalable/EntityLinker/EntitySummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include <map>
#include <memory>

namespace clang::ssaf {

/// Represents a translation unit summary in its serialized encoding.
///
/// TUSummaryEncoding holds entity summary data in a format-specific encoding
/// that can be manipulated by the entity linker without deserializing the
/// full EntitySummary objects. This enables efficient entity ID patching
/// during the linking process.
class TUSummaryEncoding {
  friend class EntityLinker;
  friend class SerializationFormat;
  friend class TestFixture;

  // The namespace identifying this translation unit.
  BuildNamespace TUNamespace;

  // Maps entity names to their unique identifiers within this TU.
  EntityIdTable IdTable;

  // Maps entity IDs to their linkage properties (None, Internal, External).
  std::map<EntityId, EntityLinkage> LinkageTable;

  // Encoded summary data organized by summary type and entity ID.
  std::map<SummaryName,
           std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
      Data;

public:
  explicit TUSummaryEncoding(BuildNamespace TUNamespace)
      : TUNamespace(std::move(TUNamespace)) {}
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_TUSUMMARYENCODING_H
