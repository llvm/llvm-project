//===- PointerFlowAnalysis.h ------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines PointerFlowAnalysisResult — the whole-program pointer-flow graph
// aggregated from per-translation-unit EdgeSet summaries.
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

constexpr llvm::StringLiteral PointerFlowAnalysisResultName =
    "PointerFlowAnalysisResult";

/// The whole-program pointer-flow graph. Each directed edge 'src -> dest'
/// records a static assignment site where 'src' (the LHS / assignee) is
/// assigned the value of 'dest' (the RHS / assigned value).
///
/// This edge direction matches property-propagation direction: if a property
/// (such as a bounds requirement) holds for 'src', it must also hold for
/// 'dest', because 'dest' supplies the value that 'src' will hold.
struct PointerFlowAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(PointerFlowAnalysisResultName.str());
  }

  std::map<EntityId, EdgeSet> Edges;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_POINTERFLOW_POINTERFLOWANALYSIS_H
