//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalyzerUnusedProgramStateRefCheck.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {
AST_MATCHER(VarDecl, isUnusedLocal) {
  return !Node.isReferenced() && !Node.hasAttr<UnusedAttr>();
}
} // namespace

static bool isUnusedProgramStateRef(const BindingDecl *B) {
  if (B->isReferenced())
    return false;
  const auto *TT = B->getType()->getAs<TypedefType>();
  return TT && TT->getDecl()->getName() == "ProgramStateRef";
}

static constexpr llvm::StringLiteral DecompID = "decomp";
static constexpr llvm::StringLiteral VarID = "var";

void AnalyzerUnusedProgramStateRefCheck::registerMatchers(MatchFinder *Finder) {
  auto ProgramStateRefType =
      qualType(hasDeclaration(namedDecl(hasName("ProgramStateRef"))));

  Finder->addMatcher(varDecl(hasLocalStorage(), isUnusedLocal(),
                             hasType(ProgramStateRefType),
                             unless(anyOf(parmVarDecl(), isInstantiated())))
                         .bind(VarID),
                     this);

  Finder->addMatcher(decompositionDecl(hasAnyBinding(bindingDecl(
                                           hasType(ProgramStateRefType))))
                         .bind(DecompID),
                     this);
}

void AnalyzerUnusedProgramStateRefCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>(VarID)) {
    diag(VD->getLocation(), "unused 'ProgramStateRef' variable %0") << VD;
    return;
  }

  // Structured binding. `[[maybe_unused]]` applies to the whole decomposition
  // rather than the individual bindings, so it is checked here.
  const auto *Decomp = Result.Nodes.getNodeAs<DecompositionDecl>(DecompID);
  if (Decomp->hasAttr<UnusedAttr>())
    return;

  // Only flag when *every* binding is an unused `ProgramStateRef`.
  if (llvm::all_of(Decomp->bindings(), isUnusedProgramStateRef))
    diag(Decomp->getLocation(), "unused 'ProgramStateRef' structured binding");
}

} // namespace clang::tidy::llvm_check
