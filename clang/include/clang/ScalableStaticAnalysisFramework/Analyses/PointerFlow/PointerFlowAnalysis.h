//===- PointerFlowAnalysis.h ------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines
// - PointerFlowAnalysisResult---the plain PointerFlow info collected from
//   the whole program.
// - PointerFlowReachableAnalysisResult---the set of reachable pointers
//   in the pointer flow graph from a provided starting set.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/StringRef.h"
#include <map>

namespace clang::ssaf {

constexpr llvm::StringLiteral PointerFlowAnalysisResultName =
    "PointerFlowAnalysisResult";
constexpr llvm::StringLiteral UnsafeBufferReachableAnalysisResultName =
    "UnsafeBufferReachableAnalysisResult";

struct PointerFlowAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(PointerFlowAnalysisResultName.str());
  }

  std::map<EntityId, EdgeSet> Edges;
};

struct UnsafeBufferReachableAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(UnsafeBufferReachableAnalysisResultName.str());
  }

  std::map<EntityId, EntityPointerLevelSet> Reachables;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H
