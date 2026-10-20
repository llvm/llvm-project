//===- NullabilitySafetyAnalysis.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whole-program nullability evidence and inference.
//
// NullabilityEvidenceAnalysisResult is the union of every contributor's
// NullabilitySafety summary.
//
// NullabilityInferenceAnalysisResult combines it with the pointer-flow graph.
// Nonnull is a must-property: an entity is inferred Nonnull only when every
// value stored to it is proven non-null. Each store contributes one piece of
// evidence: Nonnull (proven); Nullable, MaybeNull, or Unknown (each a veto);
// or Conditional on the entities it was copied from (proven once all of those
// are). Nonnull is the least fixpoint over the Conditional dependencies, so a
// cycle with no proven store is not inferred. Pointer-flow edges only
// propagate the Nullable and MaybeNull vetoes to assignees; NullableReachable
// records the entities they reach, for reporting.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_NULLABILITYSAFETY_NULLABILITYSAFETYANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_NULLABILITYSAFETY_NULLABILITYSAFETYANALYSIS_H

#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/StringRef.h"

namespace clang::ssaf {

constexpr llvm::StringLiteral NullabilityEvidenceAnalysisResultName =
    "NullabilityEvidenceAnalysisResult";
constexpr llvm::StringLiteral NullabilityInferenceAnalysisResultName =
    "NullabilityInferenceAnalysisResult";

struct NullabilityEvidenceAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(NullabilityEvidenceAnalysisResultName.str());
  }

  EntityPointerLevelSet NonnullEvidence;
  EntityPointerLevelSet NullableEvidence;
  EntityPointerLevelSet AllReturnsNonnull;
  EntityPointerLevelSet MaybeNullEvidence;
  EntityPointerLevelSet UnknownEvidence;
  EdgeSet ConditionalEvidence;
};

struct NullabilityInferenceAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(NullabilityInferenceAnalysisResultName.str());
  }

  EntityPointerLevelSet Nonnull;
  EntityPointerLevelSet Nullable;
  EntityPointerLevelSet NullableReachable;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_NULLABILITYSAFETY_NULLABILITYSAFETYANALYSIS_H
