//===- UnsafeBufferUsageBuilder.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEBUILDER_H

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include <memory>

namespace clang::ssaf {
class UnsafeBufferUsageTUSummaryBuilder : public TUSummaryBuilder {
public:
  static EntityPointerLevel buildEntityPointerLevel(EntityId Entity,
                                                    unsigned PointerLevel) {
    return {Entity, PointerLevel};
  }

  static std::unique_ptr<UnsafeBufferUsageEntitySummary>
  buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet &&UnsafeBuffers) {
    return std::make_unique<UnsafeBufferUsageEntitySummary>(
        UnsafeBufferUsageEntitySummary(std::move(UnsafeBuffers)));
  }
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEBUILDER_H
