//===- SummaryData.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for all whole-program analysis data built from
// LUSummary data. Carries no query API — all analysis-specific methods live
// on concrete subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYDATA_SUMMARYDATA_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYDATA_SUMMARYDATA_H

namespace clang::ssaf {

/// Base class for whole-program analysis data.
class SummaryData {
public:
  virtual ~SummaryData() = default;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYDATA_SUMMARYDATA_H
