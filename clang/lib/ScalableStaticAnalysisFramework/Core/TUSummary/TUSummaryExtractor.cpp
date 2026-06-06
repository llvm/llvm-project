//===- TUSummaryExtractor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include <optional>

using namespace clang;
using namespace ssaf;

static EntityLinkageType getLinkageForDecl(const Decl *D) {
  // Per the C++ standard, function parameter and return entities have no linkage.
  // For the purpose of relating them across translation units, we assign them
  // the linkage of their enclosing function.=
  if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    if (const auto *FD =
            dyn_cast_or_null<FunctionDecl>(PVD->getParentFunctionOrMethod()))
      return getLinkageForDecl(FD);
    return EntityLinkageType::None;
  }

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

std::optional<EntityId> TUSummaryExtractor::addEntity(const NamedDecl *D) {
  auto Name = getEntityName(D);
  if (!Name)
    return std::nullopt;
  return SummaryBuilder.addEntity(*Name, getLinkageForDecl(D));
}

std::optional<EntityId>
TUSummaryExtractor::addEntityForReturn(const FunctionDecl *FD) {
  auto Name = getEntityNameForReturn(FD);
  if (!Name)
    return std::nullopt;
  return SummaryBuilder.addEntity(*Name, getLinkageForDecl(FD));
}
