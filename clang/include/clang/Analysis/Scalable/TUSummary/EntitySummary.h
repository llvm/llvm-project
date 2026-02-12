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
#include "llvm/Support/ExtensibleRTTI.h"

namespace clang::ssaf {

/// Base class for analysis-specific summary data.
class EntitySummary : public llvm::RTTIExtends<EntitySummary, llvm::RTTIRoot> {
public:
  virtual ~EntitySummary() = default;
  virtual SummaryName getSummaryName() const = 0;

  static char ID; // For RTTIExtends.
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_ENTITYSUMMARY_H
