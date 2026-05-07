//===- PointerFlowAnalysis.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines PointerFlowAnalysisResult, the whole-program analysis result type
// for PointerFlowAnalysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/StringRef.h"
#include <map>

namespace clang::ssaf {

inline constexpr llvm::StringLiteral PointerFlowAnalysisResultName =
    "PointerFlowAnalysisResult";

struct PointerFlowAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(PointerFlowAnalysisResultName.str());
  }

  /// Whole-program map from EntityIds to their EdgeSets.
  std::map<EntityId, EdgeSet> Edges;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H
