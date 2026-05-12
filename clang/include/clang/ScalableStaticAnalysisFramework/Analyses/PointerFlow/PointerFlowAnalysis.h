//===- PointerFlowAnalysis.h ------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines
// - PointerFlowAnalysisResult
//     - the plain PointerFlow info collected from the whole program.
// - PointerFlowReachableAnalysisResult
//     - the set of reachable pointers in the pointer flow graph from a provided
//       starting set.
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

/// A PointerFlowAnalysisResult is a set of pointer-flow edges, i.e.,
/// a pointer-flow graph. A directed edge src -> dest corresponds to an
/// assignment (of any of various kinds, e.g., assignment operator or
/// argument-passing) of pointer dest to pointer src in the source code.
/// The edge's direction is the opposite of how pointer values flow. This
/// is because PointerFlowAnalysisResult is used for analyzing property
/// propagation between pointers. For an assignment `src = dest`, the
/// propagation works such that if `src` has a property, `dest` must also
/// have that property; otherwise, the property would not be preserved
/// across the assignment.
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
