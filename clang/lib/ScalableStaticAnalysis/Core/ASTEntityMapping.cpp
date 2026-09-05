//===- ASTMapping.cpp - AST to SSAF Entity mapping ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for mapping AST declarations to SSAF entities.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/UnifiedSymbolResolution/USRGeneration.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::ssaf {

std::optional<EntityName> getEntityName(const Decl *D) {
  if (!D)
    return std::nullopt;

  if (D->isImplicit())
    return std::nullopt;

  if (isa<FunctionDecl>(D) && cast<FunctionDecl>(D)->getBuiltinID())
    return std::nullopt;

  if (!isa<FunctionDecl, ParmVarDecl, VarDecl, FieldDecl, RecordDecl>(D))
    return std::nullopt;

  llvm::SmallString<16> Suffix;
  const Decl *USRDecl = D;

  // For parameters, use the parent function's USR with parameter index as
  // suffix
  if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    const auto *FD =
        dyn_cast_or_null<FunctionDecl>(PVD->getParentFunctionOrMethod());
    if (!FD)
      return std::nullopt;
    USRDecl = FD;

    const auto ParamIdx = PVD->getFunctionScopeIndex();
    llvm::raw_svector_ostream OS(Suffix);
    // Parameter uses function's USR with 1-based index as suffix
    OS << (ParamIdx + 1);
  }

  llvm::SmallString<128> USRBuf;
  if (clang::index::generateUSRForDecl(USRDecl, USRBuf))
    return std::nullopt;

  if (USRBuf.empty())
    return std::nullopt;

  return EntityName(USRBuf.str(), Suffix, {});
}

std::optional<EntityName> getEntityNameForReturn(const FunctionDecl *FD) {
  if (!FD)
    return std::nullopt;

  if (FD->isImplicit())
    return std::nullopt;

  if (FD->getBuiltinID())
    return std::nullopt;

  llvm::SmallString<128> USRBuf;
  if (clang::index::generateUSRForDecl(FD, USRBuf)) {
    return std::nullopt;
  }

  if (USRBuf.empty())
    return std::nullopt;

  return EntityName(USRBuf.str(), /*Suffix=*/"0", /*Namespace=*/{});
}

EntityLinkageType getLinkageForDecl(const Decl *D) {
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
    if (const auto *FD = llvm::dyn_cast_or_null<FunctionDecl>(
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

std::optional<EntityName>
getQualifiedEntityName(const Decl *D, const NestedBuildNamespace &TUNamespace,
                       const NestedBuildNamespace &LUNamespace) {
  std::optional<EntityName> Name = getEntityName(D);
  if (!Name)
    return std::nullopt;
  return Name->makeQualified(resolveNamespace(
      LUNamespace, TUNamespace, /*EntityNamespace=*/{}, getLinkageForDecl(D)));
}

std::optional<EntityName>
getQualifiedEntityNameForReturn(const FunctionDecl *FD,
                                const NestedBuildNamespace &TUNamespace,
                                const NestedBuildNamespace &LUNamespace) {
  std::optional<EntityName> Name = getEntityNameForReturn(FD);
  if (!Name)
    return std::nullopt;
  return Name->makeQualified(resolveNamespace(
      LUNamespace, TUNamespace, /*EntityNamespace=*/{}, getLinkageForDecl(FD)));
}

} // namespace clang::ssaf
