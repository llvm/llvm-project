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

namespace clang {
namespace tidy {
namespace llvm_check {

void PreferRegisterOverUnsignedCheck::registerMatchers(MatchFinder *Finder) {
  auto RegisterClassMatch = hasType(
      cxxRecordDecl(hasName("::llvm::Register")).bind("registerClassDecl"));

  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               valueDecl(allOf(
                   hasType(qualType(isUnsignedInteger()).bind("varType")),
                   varDecl(hasInitializer(exprWithCleanups(
                               has(implicitCastExpr(has(cxxMemberCallExpr(
                                   allOf(on(RegisterClassMatch),
                                         has(memberExpr(hasDeclaration(
                                             cxxConversionDecl())))))))))))
                       .bind("var")))),
      this);
}

void PreferRegisterOverUnsignedCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VarType = Result.Nodes.getNodeAs<QualType>("varType");
  const auto *UserVarDecl = Result.Nodes.getNodeAs<VarDecl>("var");

  StringRef Replacement = "llvm::Register";
  const DeclContext *Context = UserVarDecl->getDeclContext();
  while (Context) {
    if (const auto *Namespace = dyn_cast<NamespaceDecl>(Context))
      if (isa<TranslationUnitDecl>(Namespace->getDeclContext()) &&
          Namespace->getName() == "llvm")
        Replacement = "Register";
    for (const auto *UsingDirective : Context->using_directives()) {
      const NamespaceDecl *Namespace = UsingDirective->getNominatedNamespace();
      if (isa<TranslationUnitDecl>(Namespace->getDeclContext()) &&
          Namespace->getName() == "llvm")
        Replacement = "Register";
    }
    Context = Context->getParent();
  }
  diag(UserVarDecl->getLocation(),
       "variable %0 declared as %1; use '%2' instead")
      << UserVarDecl << *VarType << Replacement
      << FixItHint::CreateReplacement(
             UserVarDecl->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
             Replacement);
}

} // namespace llvm_check
} // namespace tidy
} // namespace clang
