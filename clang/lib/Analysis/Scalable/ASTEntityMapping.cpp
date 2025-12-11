//===- ASTMapping.cpp - AST to SSAF Entity mapping --------------*- C++ -*-===//
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

#include "clang/Analysis/Scalable/ASTEntityMapping.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/SmallString.h"

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

} // namespace clang::ssaf
