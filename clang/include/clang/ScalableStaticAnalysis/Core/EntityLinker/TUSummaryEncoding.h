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

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_TUSUMMARYENCODING_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_TUSUMMARYENCODING_H

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

class StaticLibraryCreateCLI;

/// Represents a translation unit summary in its serialized encoding.
///
/// TUSummaryEncoding holds entity summary data in a format-specific encoding
/// that can be manipulated by the entity linker without deserializing the
/// full EntitySummary objects. This enables efficient entity ID patching
/// during the linking process.
class TUSummaryEncoding {
  friend class EntityLinker;
  friend class SerializationFormat;
  friend class StaticLibrary;
  friend class StaticLibraryCreateCLI;
  friend class TestFixture;

  // Target triple of the translation unit.
  llvm::Triple TargetTriple;

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
  TUSummaryEncoding(llvm::Triple TargetTriple, BuildNamespace TUNamespace)
      : TargetTriple(std::move(TargetTriple)),
        TUNamespace(std::move(TUNamespace)) {}

  const llvm::Triple &getTargetTriple() const { return TargetTriple; }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_TUSUMMARYENCODING_H
