//===- EntityLookup.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps the declarations of the translation unit a transformation runs on to
// the entities of the whole-program analysis result.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_ENTITYLOOKUP_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_ENTITYLOOKUP_H

#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include <map>
#include <optional>

namespace clang {
class Decl;
class FunctionDecl;
} // namespace clang

namespace clang::ssaf {

class SSAFOptions;
class WPASuite;

/// Looks up the EntityId that a WPASuite gives a declaration of the
/// translation unit identified by SSAFOptions::CompilationUnitId, linked into
/// SSAFOptions::LinkUnitId.
class EntityLookup {
public:
  EntityLookup(const WPASuite &Suite, const SSAFOptions &Opts);

  /// Returns the EntityId of \p D, or std::nullopt if \p D is not an entity or
  /// the suite does not contain it.
  std::optional<EntityId> getEntityId(const Decl *D) const;

  /// Returns the EntityId of the return value of \p FD, or std::nullopt if the
  /// return value is not an entity or the suite does not contain it.
  std::optional<EntityId> getEntityIdForReturn(const FunctionDecl *FD) const;

private:
  std::optional<EntityId> lookup(const std::optional<EntityName> &Name) const;

  std::map<EntityName, EntityId> Ids;
  NestedBuildNamespace TUNamespace;
  NestedBuildNamespace LUNamespace;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_ENTITYLOOKUP_H
