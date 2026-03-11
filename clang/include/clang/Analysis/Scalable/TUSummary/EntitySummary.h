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
#include <type_traits>

namespace clang::ssaf {

/// Base class for analysis-specific summary data.
class EntitySummary {
public:
  virtual ~EntitySummary() = default;
  virtual SummaryName getSummaryName() const = 0;
};

template <typename Derived>
using DerivesFromEntitySummary =
    std::enable_if_t<std::is_base_of_v<EntitySummary, Derived>>;

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_ENTITYSUMMARY_H
