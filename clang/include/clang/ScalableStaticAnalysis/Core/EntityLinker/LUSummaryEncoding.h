//===- LUSummaryEncoding.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LUSummaryEncoding class, which represents a link unit
// summary in its serialized, format-specific encoding.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LUSUMMARYENCODING_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LUSUMMARYENCODING_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntitySummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "llvm/TargetParser/Triple.h"
#include <map>
#include <memory>

namespace clang::ssaf {

/// Represents a link unit summary in its serialized encoding.
///
/// LUSummaryEncoding holds the combined entity summary data from multiple
/// translation units in a format-specific encoding. It is produced by the
/// entity linker and contains deduplicated and patched entity summaries.
class LUSummaryEncoding {
  friend class EntityLinker;
  friend class MultiArchSharedLibrary;
  friend class SerializationFormat;
  friend class TestFixture;

  // Target triple of the link unit.
  llvm::Triple TargetTriple;

  // The namespace identifying this link unit.
  NestedBuildNamespace LUNamespace;

  // Maps entity names to their unique identifiers within this link unit.
  EntityIdTable IdTable;

  // Maps entity IDs to their linkage properties.
  std::map<EntityId, EntityLinkage> LinkageTable;

  // Encoded summary data organized by summary type and entity ID.
  std::map<SummaryName,
           std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
      Data;

public:
  LUSummaryEncoding(llvm::Triple TargetTriple, NestedBuildNamespace LUNamespace)
      : TargetTriple(std::move(TargetTriple)),
        LUNamespace(std::move(LUNamespace)) {}
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LUSUMMARYENCODING_H
