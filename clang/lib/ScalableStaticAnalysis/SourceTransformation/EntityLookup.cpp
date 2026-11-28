//===- EntityLookup.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/EntityLookup.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"

using namespace clang;
using namespace clang::ssaf;

EntityLookup::EntityLookup(const WPASuite &Suite, const SSAFOptions &Opts)
    : TUNamespace(
          NestedBuildNamespace::makeCompilationUnit(Opts.CompilationUnitId)),
      LUNamespace(NestedBuildNamespace::makeLinkUnit(Opts.LinkUnitId)) {
  Suite.getIdTable().forEach(
      [this](const EntityName &Name, EntityId Id) { Ids.emplace(Name, Id); });
}

std::optional<EntityId> EntityLookup::getEntityId(const Decl *D) const {
  return lookup(getQualifiedEntityName(D, TUNamespace, LUNamespace));
}

std::optional<EntityId>
EntityLookup::getEntityIdForReturn(const FunctionDecl *FD) const {
  return lookup(getQualifiedEntityNameForReturn(FD, TUNamespace, LUNamespace));
}

std::optional<EntityId>
EntityLookup::lookup(const std::optional<EntityName> &Name) const {
  if (!Name)
    return std::nullopt;
  auto It = Ids.find(*Name);
  if (It == Ids.end())
    return std::nullopt;
  return It->second;
}
