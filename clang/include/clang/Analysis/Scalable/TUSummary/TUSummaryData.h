//===- TUSummaryData.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYDATA_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYDATA_H

#include "clang/Analysis/Scalable/Model/SummaryName.h"

namespace clang::ssaf {

/// Base class for analysis-specific summary data.
class TUSummaryData {
private:
  /// Name of the summary.
  SummaryName Summary;

protected:
  TUSummaryData(SummaryName Summary) : Summary(std::move(Summary)) {}

public:
  SummaryName getSummaryName() const { return Summary; }

  virtual ~TUSummaryData() = default;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYDATA_H
