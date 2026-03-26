//===- UnsafeBufferUsageExtractor.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEBUILDER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEBUILDER_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace clang::ssaf {
class UnsafeBufferUsageTUSummaryExtractor : public TUSummaryExtractor {
public:
  UnsafeBufferUsageTUSummaryExtractor(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {}

  static EntityPointerLevel buildEntityPointerLevel(EntityId Entity,
                                                    unsigned PointerLevel) {
    return {Entity, PointerLevel};
  }

  EntityId addEntity(EntityName EN) { return SummaryBuilder.addEntity(EN); }

  std::unique_ptr<UnsafeBufferUsageEntitySummary>
  extractEntitySummary(const Decl *Contributor, ASTContext &Ctx,
                       llvm::Error &Error);

  void HandleTranslationUnit(ASTContext &Ctx) override;
};
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGEBUILDER_H
