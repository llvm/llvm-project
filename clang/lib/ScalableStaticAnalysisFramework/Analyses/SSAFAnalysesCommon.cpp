//===- SSAFAnalysesCommon.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

llvm::Error clang::ssaf::makeEntityNameErr(clang::ASTContext &Ctx,
                                           const clang::NamedDecl *D) {
  return makeErrAtNode(Ctx, D, "failed to create entity name for %s",
                       D->getNameAsString().data());
}
