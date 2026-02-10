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

namespace clang::ssaf {

class EntityId;
class EntityLinkage;
class EntityName;
class EntitySummary;
class TUSummary;

class TUSummaryBuilder {
public:
  explicit TUSummaryBuilder(TUSummary &Summary) : Summary(Summary) {}

  /// Add an entity to the summary and return its EntityId.
  /// If the entity already exists, returns the existing ID (idempotent).
  EntityId addEntity(const EntityName &E, const EntityLinkage &Linkage);

  /// Add analysis-specific fact data for an entity.
  /// Precondition: The ContributingEntity must have been added via addEntity().
  void addFact(EntityId ContributingEntity,
               std::unique_ptr<EntitySummary> NewData);

private:
  TUSummary &Summary;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
