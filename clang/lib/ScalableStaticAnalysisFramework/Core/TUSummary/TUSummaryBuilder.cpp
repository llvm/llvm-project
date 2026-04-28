//===- TUSummaryBuilder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include <memory>
#include <optional>
#include <utility>

using namespace clang;
using namespace ssaf;

static EntityLinkageType getLinkageForDecl(const Decl *D) {
  const auto *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return EntityLinkageType::None;

  switch (ND->getFormalLinkage()) {
  case Linkage::Invalid: {
    llvm_unreachable("Shouldn't be invalid");
  }
  case Linkage::None:
    return EntityLinkageType::None;
  case Linkage::Internal:
    return EntityLinkageType::Internal;
  case Linkage::UniqueExternal:
    return EntityLinkageType::Internal;
  case Linkage::VisibleNone:
    return EntityLinkageType::Internal;
  case Linkage::Module:
    return EntityLinkageType::External;
  case Linkage::External:
    return EntityLinkageType::External;
  }
  llvm_unreachable("Unhandled clang::Linkage kind");
}

EntityId TUSummaryBuilder::addEntityImpl(const EntityName &Name,
                                         const Decl *D) {
  EntityId Id = Summary.IdTable.getId(Name);
  Summary.LinkageTable.try_emplace(Id, getLinkageForDecl(D));
  return Id;
}

std::optional<EntityId> TUSummaryBuilder::addEntity(const NamedDecl *D) {
  auto Name = getEntityName(D);
  if (!Name)
    return std::nullopt;
  return addEntityImpl(*Name, D);
}

std::optional<EntityId>
TUSummaryBuilder::addEntityForReturn(const FunctionDecl *FD) {
  auto Name = getEntityNameForReturn(FD);
  if (!Name)
    return std::nullopt;
  return addEntityImpl(*Name, FD);
}

std::pair<EntitySummary *, bool>
TUSummaryBuilder::addSummaryImpl(EntityId Entity,
                                 std::unique_ptr<EntitySummary> &&Data) {
  auto &EntitySummaries = Summary.Data[Data->getSummaryName()];
  auto [It, Inserted] = EntitySummaries.try_emplace(Entity, std::move(Data));
  return {It->second.get(), Inserted};
}
