//===- TUSummaryBuilder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include <memory>
#include <utility>

using namespace clang;
using namespace ssaf;

EntityId TUSummaryBuilder::addEntity(const EntityName &EN,
                                     EntityLinkage Linkage) {
  EntityId Id = Summary.IdTable.getId(EN);
  [[maybe_unused]] const EntityLinkage &Existing =
      Summary.LinkageTable.try_emplace(Id, Linkage).first->second;
  // An entity's linkage type must be stable across its redeclarations within a
  // TU. The other symbol properties (binding, visibility, definition kind) may
  // legitimately differ between a declaration and a definition of the same
  // entity; reconciling those is the linker's responsibility, so we keep the
  // first occurrence here and only assert linkage-type stability.
  assert(Existing.getLinkage() == Linkage.getLinkage());
  return Id;
}

std::pair<EntitySummary *, bool>
TUSummaryBuilder::addSummaryImpl(EntityId Entity,
                                 std::unique_ptr<EntitySummary> &&Data) {
  auto &EntitySummaries = Summary.Data[Data->getSummaryName()];
  auto [It, Inserted] = EntitySummaries.try_emplace(Entity, std::move(Data));
  return {It->second.get(), Inserted};
}
