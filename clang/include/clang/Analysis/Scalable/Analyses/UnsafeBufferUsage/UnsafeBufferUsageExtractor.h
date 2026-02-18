//===- UnsafeBufferUsageExtractor.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_EXTRACTOR_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_EXTRACTOR_H

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsageBuilder.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummaryExtractor.h"
#include <memory>

namespace clang::ssaf {
class UnsafeBufferUsageTUSummaryExtractor : public TUSummaryExtractor {

  UnsafeBufferUsageTUSummaryBuilder &getBuilder() {
    return static_cast<UnsafeBufferUsageTUSummaryBuilder &>(SummaryBuilder);
  }

public:
  explicit UnsafeBufferUsageTUSummaryExtractor(
      UnsafeBufferUsageTUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {}

  // FIXME: need some general traversal in the Base class
  std::unique_ptr<UnsafeBufferUsageEntitySummary>
  extractEntitySummary(EntityId Contributor, const Decl *ContributorDefn,
                       ASTContext &Ctx);
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ANALYSES_UNSAFEBUFFERUSAGE_EXTRACTOR_H