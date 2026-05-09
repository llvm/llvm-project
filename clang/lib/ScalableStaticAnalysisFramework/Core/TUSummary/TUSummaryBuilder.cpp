//===- TUSummaryBuilder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include <memory>
#include <utility>

using namespace clang;
using namespace ssaf;

EntityId TUSummaryBuilder::addEntity(const EntityName &EN,
                                     EntityLinkageType Linkage) {
  EntityId Id = Summary.IdTable.getId(EN);
  [[maybe_unused]] EntityLinkageType Link =
      Summary.LinkageTable.try_emplace(Id, Linkage).first->second.getLinkage();
  // Even if we had in the past a linkage, that must bee the same as we set now.
  assert(Link == Linkage);
  return Id;
}

std::pair<EntitySummary *, bool>
TUSummaryBuilder::addSummaryImpl(EntityId Entity,
                                 std::unique_ptr<EntitySummary> &&Data) {
  auto &EntitySummaries = Summary.Data[Data->getSummaryName()];
  auto [It, Inserted] = EntitySummaries.try_emplace(Entity, std::move(Data));
  return {It->second.get(), Inserted};
}
