//===- VirtualMethodFamily.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILY_H

#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <tuple>
#include <vector>

namespace clang::ssaf {

struct VirtualMethodSummary final : public EntitySummary {
  static constexpr llvm::StringLiteral Name = "VirtualMethod";

  static SummaryName summaryName() { return SummaryName(Name.str()); }

  SummaryName getSummaryName() const override { return summaryName(); }

  /// EntityIds of each ParmVarDecl, in source order.
  std::vector<EntityId> ParamEntities;

  /// EntityId of the synthetic return-slot entity for this method.
  std::optional<EntityId> ReturnEntity;

  /// The result of \c CXXMethodDecl::overridden_methods().
  std::vector<EntityId> OverriddenMethods;

  bool operator==(const VirtualMethodSummary &Other) const {
    return std::tie(ParamEntities, ReturnEntity, OverriddenMethods) ==
           std::tie(Other.ParamEntities, Other.ReturnEntity,
                    Other.OverriddenMethods);
  }

  bool operator!=(const VirtualMethodSummary &Other) const {
    return !(*this == Other);
  }
};

struct VirtualMethodFamilyAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("VirtualMethodFamilyAnalysisResult");
  }

  /// Maps each parameter or return slot to the ID of the family it belongs to.
  /// The family ID is the smallest slot ID in the family.
  llvm::DenseMap<EntityId, EntityId> RetAndParamData;

  bool operator==(const VirtualMethodFamilyAnalysisResult &Other) const {
    return RetAndParamData == Other.RetAndParamData;
  }

  bool operator!=(const VirtualMethodFamilyAnalysisResult &Other) const {
    return !(*this == Other);
  }
};

/// Prints \p R as one "<param/return id> -> <family id>" line per entry,
/// ordered by the param/return id so that the output is stable across runs.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const VirtualMethodFamilyAnalysisResult &R);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILY_H
