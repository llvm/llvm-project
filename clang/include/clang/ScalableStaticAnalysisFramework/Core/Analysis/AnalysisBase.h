//===- AnalysisBase.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal common base for SummaryAnalysisBase and DerivedAnalysisBase.
// Carries the identity (analysisName()) and dependency list
// (dependencyNames()) shared by every analysis regardless of kind.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_ANALYSIS_ANALYSISBASE_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_ANALYSIS_ANALYSISBASE_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/AnalysisName.h"
#include <vector>

namespace clang::ssaf {

class AnalysisDriver;
class SummaryAnalysisBase;
class DerivedAnalysisBase;

/// Minimal common base for both analysis kinds.
///
/// Not subclassed directly — use SummaryAnalysis<...> or
/// DerivedAnalysis<...> instead.
class AnalysisBase {
  friend class AnalysisDriver;
  friend class SummaryAnalysisBase;
  friend class DerivedAnalysisBase;

  enum class Kind { Summary, Derived };
  Kind TheKind;

protected:
  explicit AnalysisBase(Kind K) : TheKind(K) {}

public:
  virtual ~AnalysisBase() = default;

  /// Name of this analysis. Equal to ResultT::analysisName() in both typed
  /// intermediates.
  virtual AnalysisName analysisName() const = 0;

  /// AnalysisNames of all AnalysisResult dependencies.
  virtual const std::vector<AnalysisName> &dependencyNames() const = 0;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_ANALYSIS_ANALYSISBASE_H
