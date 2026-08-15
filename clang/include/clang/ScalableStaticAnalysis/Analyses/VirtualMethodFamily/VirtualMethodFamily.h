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

/// Prints \p S as
/// "VirtualMethodSummary { params=[...], return=..., overridden=[...] }".
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const VirtualMethodSummary &S);

struct VirtualMethodFamilyAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("VirtualMethodFamilyAnalysisResult");
  }

  struct Data {
    /// Represents the ID of the family the given parameter or return ID
    /// corresponds to.
    /// Right now, this ID is the "smallest" ID of the method in
    /// the overloading set.
    EntityId FamilyId;

    /// The virtual method IDs of the param/return IDs it correspond to.
    /// Basically, for "param" in "fun(param)" it will be "fun".
    EntityId OwnerMethodId;
  };
  llvm::DenseMap<EntityId, Data> RetAndParamData;

  friend bool operator==(const Data &L, const Data &R) {
    return std::tie(L.FamilyId, L.OwnerMethodId) ==
           std::tie(R.FamilyId, R.OwnerMethodId);
  }
  friend bool operator!=(const Data &L, const Data &R) { return !(L == R); }

  bool operator==(const VirtualMethodFamilyAnalysisResult &Other) const {
    return RetAndParamData == Other.RetAndParamData;
  }

  bool operator!=(const VirtualMethodFamilyAnalysisResult &Other) const {
    return !(*this == Other);
  }
};

/// Prints \p D as "{family=EntityId(1), owner=EntityId(2)}".
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const VirtualMethodFamilyAnalysisResult::Data &D);

/// Prints \p R as one "<param/return id> -> <data>" line per entry, ordered by
/// the param/return id so that the output is stable across runs.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const VirtualMethodFamilyAnalysisResult &R);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILY_H
