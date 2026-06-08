//===- TUSummaryExtractor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace clang;
using namespace ssaf;

static EntityLinkageType getLinkageForDecl(const Decl *D) {
  const auto *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return EntityLinkageType::None;

  // Parameters have no linkage in C++, but SSAF needs them to inherit
  // the external linkage from their parent functions.
  // Here is why:
  //   SSAF treats parameters as entities and may not always associate them back
  //   to their parent functions. Therefore, it needs to identify parameters of
  //   functions with external linkage across different TUs. Treating them as
  //   having no linkage (as in C++) causes the same parameter in different TUs
  //   to be assigned different EntityIDs. As a result, the behavior of the
  //   parameter across multiple TUs cannot be correlated.
  if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    if (const auto *FD = llvm::dyn_cast<FunctionDecl>(
            PVD->getParentFunctionOrMethod())) {
      return getLinkageForDecl(FD);
    }
  }

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
