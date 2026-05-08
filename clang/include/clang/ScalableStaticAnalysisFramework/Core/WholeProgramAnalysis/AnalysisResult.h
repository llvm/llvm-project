//===- AnalysisResult.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for all whole-program analysis results produced by AnalysisDriver.
// Concrete subclasses carry a static analysisName().
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISRESULT_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISRESULT_H

namespace clang::ssaf {

/// Base class for whole-program analysis results.
///
/// Concrete subclasses must provide:
///   static AnalysisName analysisName();
class AnalysisResult {
public:
  virtual ~AnalysisResult() = default;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISRESULT_H
