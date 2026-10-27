//===- NullabilitySafety.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per-contributor nullability evidence observed by the nullability safety
// analysis: which parameters, fields and function returns received a
// provably non-null value, a value that is null on every path, or one that is
// null on only some paths, and which functions return a non-null pointer on
// every path.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_NULLABILITYSAFETY_NULLABILITYSAFETY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_NULLABILITYSAFETY_NULLABILITYSAFETY_H

#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"

namespace clang::ssaf {
class NullabilitySafetyEntitySummary final : public EntitySummary {
  EntityPointerLevelSet NonnullEvidence;
  EntityPointerLevelSet NullableEvidence;
  EntityPointerLevelSet AllReturnsNonnull;
  EntityPointerLevelSet MaybeNullEvidence;
  EntityPointerLevelSet UnknownEvidence;
  EdgeSet ConditionalEvidence;

public:
  static constexpr llvm::StringLiteral Name = "NullabilitySafety";

  NullabilitySafetyEntitySummary(EntityPointerLevelSet NonnullEvidence,
                                 EntityPointerLevelSet NullableEvidence,
                                 EntityPointerLevelSet AllReturnsNonnull,
                                 EntityPointerLevelSet MaybeNullEvidence,
                                 EntityPointerLevelSet UnknownEvidence,
                                 EdgeSet ConditionalEvidence)
      : NonnullEvidence(std::move(NonnullEvidence)),
        NullableEvidence(std::move(NullableEvidence)),
        AllReturnsNonnull(std::move(AllReturnsNonnull)),
        MaybeNullEvidence(std::move(MaybeNullEvidence)),
        UnknownEvidence(std::move(UnknownEvidence)),
        ConditionalEvidence(std::move(ConditionalEvidence)) {}

  SummaryName getSummaryName() const override { return summaryName(); }

  const EntityPointerLevelSet &getNonnullEvidence() const {
    return NonnullEvidence;
  }
  const EntityPointerLevelSet &getNullableEvidence() const {
    return NullableEvidence;
  }
  const EntityPointerLevelSet &getAllReturnsNonnull() const {
    return AllReturnsNonnull;
  }
  const EntityPointerLevelSet &getMaybeNullEvidence() const {
    return MaybeNullEvidence;
  }
  const EntityPointerLevelSet &getUnknownEvidence() const {
    return UnknownEvidence;
  }
  const EdgeSet &getConditionalEvidence() const { return ConditionalEvidence; }

  bool operator==(const NullabilitySafetyEntitySummary &Other) const {
    return NonnullEvidence == Other.NonnullEvidence &&
           NullableEvidence == Other.NullableEvidence &&
           AllReturnsNonnull == Other.AllReturnsNonnull &&
           MaybeNullEvidence == Other.MaybeNullEvidence &&
           UnknownEvidence == Other.UnknownEvidence &&
           ConditionalEvidence == Other.ConditionalEvidence;
  }

  bool empty() const {
    return NonnullEvidence.empty() && NullableEvidence.empty() &&
           AllReturnsNonnull.empty() && MaybeNullEvidence.empty() &&
           UnknownEvidence.empty() && ConditionalEvidence.empty();
  }

  static SummaryName summaryName() { return SummaryName{Name.str()}; }
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_NULLABILITYSAFETY_NULLABILITYSAFETY_H
