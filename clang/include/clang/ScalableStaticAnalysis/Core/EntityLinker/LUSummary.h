//===- LUSummary.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LUSummary class, which represents a link unit summary
// containing merged and deduplicated entity summaries from multiple TUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LUSUMMARY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LUSUMMARY_H

#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "llvm/TargetParser/Triple.h"
#include <map>
#include <memory>

namespace clang::ssaf {

/// Represents a link unit (LU) summary containing merged entity summaries.
///
/// LUSummary is the result of linking multiple translation unit summaries
/// together. It contains deduplicated entities with their linkage information
/// and the merged entity summaries.
class LUSummary {
  friend class AnalysisDriver;
  friend class LUSummaryConsumer;
  friend class SerializationFormat;
  friend class TestFixture;

  /// Target triple of the link unit.
  llvm::Triple TargetTriple;

  NestedBuildNamespace LUNamespace;

  EntityIdTable IdTable;

  std::map<EntityId, EntityLinkage> LinkageTable;

  std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>
      Data;

public:
  LUSummary(llvm::Triple TargetTriple, NestedBuildNamespace LUNamespace)
      : TargetTriple(std::move(TargetTriple)),
        LUNamespace(std::move(LUNamespace)) {}

  const NestedBuildNamespace &getNamespace() const { return LUNamespace; }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LUSUMMARY_H
