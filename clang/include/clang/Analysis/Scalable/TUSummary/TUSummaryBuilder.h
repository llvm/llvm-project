//===- TUSummaryBuilder.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H

#include <memory>
#include <utility>

namespace clang::ssaf {

class EntityId;
class EntityName;
class EntitySummary;
class TUSummary;

class TUSummaryBuilder {
public:
  explicit TUSummaryBuilder(TUSummary &Summary) : Summary(Summary) {}

  /// Add an entity to the summary and return its EntityId.
  /// If the entity already exists, returns the existing ID (idempotent).
  EntityId addEntity(const EntityName &E);

  /// Associate the \p Data EntitySummary with the \p Entity.
  /// This consumes the \p Data only if \p Entity wasn't associated yet with the
  /// same kind of EntitySummary.
  /// \returns a pointer to the EntitySummary and whether it inserted or not.
  /// \note Be sure to pass exactly an expression of type
  /// \sa std::unique_ptr<EntitySummary>, otherwise the conversion operator will
  /// automatically consume the \p Data.
  std::pair<EntitySummary *, bool>
  addSummary(EntityId Entity, std::unique_ptr<EntitySummary> &&Data);

private:
  TUSummary &Summary;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
