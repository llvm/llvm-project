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
#include "llvm/ADT/StringRef.h"
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

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILY_H
