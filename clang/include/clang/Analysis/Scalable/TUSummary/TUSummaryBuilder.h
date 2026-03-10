//===- TUSummaryBuilder.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include <memory>
#include <utility>

namespace clang::ssaf {

class EntityName;
class TUSummary;

class TUSummaryBuilder {
public:
  explicit TUSummaryBuilder(TUSummary &Summary) : Summary(Summary) {}

  /// Add an entity to the summary and return its EntityId.
  /// If the entity already exists, returns the existing ID (idempotent).
  EntityId addEntity(const EntityName &E);

  /// Associate the \p Data \c EntitySummary with the \p Entity.
  /// This consumes the \p Data only if \p Entity wasn't associated yet with the
  /// same kind of \c EntitySummary.
  /// \returns a pointer to the \c EntitySummary and whether it inserted or not.
  template <typename ConcreteEntitySummary,
            DerivesFromEntitySummary<ConcreteEntitySummary> * = nullptr>
  std::pair<EntitySummary *, bool>
  addSummary(EntityId Entity, std::unique_ptr<ConcreteEntitySummary> &&Data);

private:
  TUSummary &Summary;

  std::pair<EntitySummary *, bool>
  addSummaryImpl(EntityId Entity, std::unique_ptr<EntitySummary> &&Data);
};

// Why is this a template?
//
// We use template here to avoid an implicit conversion from
// `std::unique_ptr<ConcreteEntitySummary>` to `std::unique_ptr<EntitySummary>`
// because constructing that implicit temporary would unconditionally "consume"
// the Data. This would make it impossible to recover from the call-site the
// Data you pass in even if no insertion happens.
template <typename ConcreteEntitySummary,
          DerivesFromEntitySummary<ConcreteEntitySummary> *>
std::pair<EntitySummary *, bool>
TUSummaryBuilder::addSummary(EntityId Entity,
                             std::unique_ptr<ConcreteEntitySummary> &&Data) {
  // Prepare a unique_ptr of the base type to avoid implicit conversions at the
  // call-site.
  std::unique_ptr<EntitySummary> TypeErasedData = std::move(Data);

  // This only moves (consumes) TypeErasedData if insertion happened.
  // Otherwise it doesn't touch the `TypeErasedData`.
  auto [It, Inserted] = addSummaryImpl(Entity, std::move(TypeErasedData));

  // Move it back on failue to keep the `Data` unconsumed.
  if (!Inserted) {
    Data = std::unique_ptr<ConcreteEntitySummary>(
        static_cast<ConcreteEntitySummary *>(TypeErasedData.release()));
  }
  return {It, Inserted};
}

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
