//===- UnsafeBufferUsage.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include <set>

namespace clang::ssaf {

/// An UnsafeBufferUsageEntitySummary is an immutable set of unsafe buffers, in
/// the form of EntityPointerLevel.
class UnsafeBufferUsageEntitySummary final : public EntitySummary {
  const EntityPointerLevelSet UnsafeBuffers;

  friend class UnsafeBufferUsageTUSummaryExtractor;
  friend UnsafeBufferUsageEntitySummary
      buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet);
  friend llvm::iterator_range<EntityPointerLevelSet::const_iterator>
  getUnsafeBuffers(const UnsafeBufferUsageEntitySummary &);

  UnsafeBufferUsageEntitySummary(EntityPointerLevelSet UnsafeBuffers)
      : EntitySummary(), UnsafeBuffers(std::move(UnsafeBuffers)) {}

public:
  static constexpr llvm::StringLiteral Name = "UnsafeBufferUsage";

  SummaryName getSummaryName() const override { return summaryName(); };

  bool operator==(const EntityPointerLevelSet &Other) const {
    return UnsafeBuffers == Other;
  }

  bool operator==(const UnsafeBufferUsageEntitySummary &Other) const {
    return UnsafeBuffers == Other.UnsafeBuffers;
  }

  bool empty() const { return UnsafeBuffers.empty(); }

  static SummaryName summaryName() { return SummaryName{Name.str()}; }
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGE_H
