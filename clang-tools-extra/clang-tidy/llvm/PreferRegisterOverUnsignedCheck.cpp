//===--- PreferRegisterOverUnsignedCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferRegisterOverUnsignedCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void PreferRegisterOverUnsignedCheck::registerMatchers(MatchFinder *Finder) {
  auto RegisterClassMatch = hasType(
      cxxRecordDecl(hasName("::llvm::Register")).bind("registerClassDecl"));

  Finder->addMatcher(
      traverse(TK_AsIs,
               valueDecl(hasType(qualType(isUnsignedInteger()).bind("varType")),
                         varDecl(hasInitializer(exprWithCleanups(
                                     has(implicitCastExpr(has(cxxMemberCallExpr(
                                         on(RegisterClassMatch),
                                         has(memberExpr(hasDeclaration(
                                             cxxConversionDecl()))))))))))
                             .bind("var"))),
      this);
}

void PreferRegisterOverUnsignedCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VarType = Result.Nodes.getNodeAs<QualType>("varType");
  const auto *UserVarDecl = Result.Nodes.getNodeAs<VarDecl>("var");

  bool NeedsQualification = true;
  const DeclContext *Context = UserVarDecl->getDeclContext();
  while (Context) {
    if (const auto *Namespace = dyn_cast<NamespaceDecl>(Context))
      if (isa<TranslationUnitDecl>(Namespace->getDeclContext()) &&
          Namespace->getName() == "llvm")
        NeedsQualification = false;
    for (const auto *UsingDirective : Context->using_directives()) {
      const NamespaceDecl *Namespace = UsingDirective->getNominatedNamespace();
      if (isa<TranslationUnitDecl>(Namespace->getDeclContext()) &&
          Namespace->getName() == "llvm")
        NeedsQualification = false;
    }
    Context = Context->getParent();
  }
  diag(UserVarDecl->getLocation(),
       "variable %0 declared as %1; use '%select{|llvm::}2Register' instead")
      << UserVarDecl << *VarType << NeedsQualification
      << FixItHint::CreateReplacement(
             UserVarDecl->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
             NeedsQualification ? "llvm::Register" : "Register");
}

} // namespace clang::tidy::llvm_check
