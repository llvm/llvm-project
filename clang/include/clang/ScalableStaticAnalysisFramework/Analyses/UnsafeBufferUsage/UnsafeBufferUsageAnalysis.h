//===- UnsafeBufferUsageAnalysis.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines UnsafeBufferUsageAnalysisResult, the whole-program analysis result
// type for UnsafeBufferUsageAnalysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEANALYSIS_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/StringRef.h"
#include <map>

namespace clang::ssaf {

constexpr llvm::StringLiteral UnsafeBufferUsageAnalysisResultName =
    "UnsafeBufferUsageAnalysisResult";

struct UnsafeBufferUsageAnalysisResult final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName(UnsafeBufferUsageAnalysisResultName.str());
  }

  /// Whole-program set of unsafe buffer pointers:
  std::map<EntityId, EntityPointerLevelSet> UnsafeBuffers;

  auto begin() const { return UnsafeBuffers.begin(); }
  auto end() const { return UnsafeBuffers.end(); }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEANALYSIS_H
