//===- TUSummaryExtractor.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_TUSUMMARYEXTRACTOR_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_TUSUMMARYEXTRACTOR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include <optional>

namespace clang::ssaf {
class TUSummaryBuilder;

class TUSummaryExtractor : public ASTConsumer {
public:
  explicit TUSummaryExtractor(TUSummaryBuilder &Builder)
      : SummaryBuilder(Builder) {}

  /// Creates EntityName from the Decl, registers the entity, and sets its
  /// linkage atomically.
  /// \returns the EntityId, or std::nullopt if EntityName creation fails.
  std::optional<EntityId> addEntity(const NamedDecl *D);

  /// Creates EntityName for the return value of \p FD, registers the entity,
  /// and sets its linkage atomically.
  /// \returns the EntityId, or std::nullopt if EntityName creation fails.
  std::optional<EntityId> addEntityForReturn(const FunctionDecl *FD);

protected:
  TUSummaryBuilder &SummaryBuilder;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_TUSUMMARYEXTRACTOR_H
