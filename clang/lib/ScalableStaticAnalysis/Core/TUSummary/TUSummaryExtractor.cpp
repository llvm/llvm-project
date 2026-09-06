//===- TUSummaryExtractor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace clang;
using namespace ssaf;

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

const SSAFOptions &TUSummaryExtractor::getOptions() const {
  return SummaryBuilder.getOptions();
}
