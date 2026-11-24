//===- TUSummary.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_TUSUMMARY_TUSUMMARY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_TUSUMMARY_TUSUMMARY_H

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

/// Data extracted for a given translation unit and for a given set of analyses.
class TUSummary {
  /// Target triple of the translation unit.
  llvm::Triple TargetTriple;

  /// Identifies the translation unit.
  BuildNamespace TUNamespace;

  EntityIdTable IdTable;

  std::map<EntityId, EntityLinkage> LinkageTable;

  std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>
      Data;

public:
  TUSummary(llvm::Triple TargetTriple, BuildNamespace TUNamespace)
      : TargetTriple(std::move(TargetTriple)),
        TUNamespace(std::move(TUNamespace)) {}

  friend class SerializationFormat;
  friend class TestFixture;
  friend class TUSummaryBuilder;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_TUSUMMARY_TUSUMMARY_H
