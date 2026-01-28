//===- EntitySummary.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_ENTITYSUMMARY_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_ENTITYSUMMARY_H

#include "clang/Analysis/Scalable/Model/SummaryName.h"

namespace clang::ssaf {

/// Base class for analysis-specific summary data.
class EntitySummary {
private:
  SummaryName Summary;

protected:
  EntitySummary(SummaryName Summary) : Summary(std::move(Summary)) {}

public:
  SummaryName getSummaryName() const { return Summary; }

  virtual ~EntitySummary() = default;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_ENTITYSUMMARY_H
