//===- SSAFAnalysesCommon.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Common code in SSAF analyses implementations
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_SSAFANALYSESCOMMON_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_SSAFANALYSESCOMMON_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/JSON.h"

namespace clang::ssaf {
template <typename NodeTy, typename... Ts>
llvm::Error makeErrAtNode(clang::ASTContext &Ctx, const NodeTy *N,
                          llvm::StringRef Fmt, const Ts &...Args) {
  std::string LocStr = N->getBeginLoc().printToString(Ctx.getSourceManager());
  return llvm::createStringError((Fmt + " at %s").str().c_str(), Args...,
                                 LocStr.c_str());
}

template <typename... Ts>
llvm::Error makeSawButExpectedError(const llvm::json::Value &Saw,
                                    llvm::StringRef Expected,
                                    const Ts &...ExpectedArgs) {
  std::string Fmt = ("saw %s but expected " + Expected).str();
  std::string SawStr = llvm::formatv("{0:2}", Saw).str();

  return llvm::createStringError(Fmt.c_str(), SawStr.c_str(), ExpectedArgs...);
}

template <typename DeclOrExpr> bool hasPtrOrArrType(const DeclOrExpr *E) {
  return llvm::isa<clang::PointerType, clang::ArrayType>(
      E->getType().getCanonicalType());
}

llvm::Error makeEntityNameErr(clang::ASTContext &Ctx,
                              const clang::NamedDecl *D);

/// Find all contributors in an AST.
void findContributors(ASTContext &Ctx,
                      std::vector<const NamedDecl *> &Contributors);

/// Perform "MatchAction" on each Stmt and Decl belonging to the `Contributor`.
/// \param Contributor
/// \param MatchActionRef a reference (view) to a "MatchAction"
void findMatchesIn(
    const NamedDecl *Contributor,
    llvm::function_ref<void(const DynTypedNode &)> MatchActionRef);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_SSAFANALYSESCOMMON_H
