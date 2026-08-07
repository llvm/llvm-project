//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TemplateVirtualMemberFunctionCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::portability {
namespace {
AST_MATCHER(CXXMethodDecl, isUsed) { return Node.isUsed(); }
} // namespace

void TemplateVirtualMemberFunctionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(isVirtual(),
                    ofClass(classTemplateSpecializationDecl(
                                unless(isExplicitTemplateSpecialization()))
                                .bind("specialization")),
                    unless(isUsed()), unless(isPure()),
                    unless(cxxDestructorDecl(isDefaulted())))
          .bind("method"),
      this);
}

void TemplateVirtualMemberFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *ImplicitSpecialization =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("specialization");
  const auto *MethodDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("method");

  diag(MethodDecl->getLocation(),
       "unspecified virtual member function instantiation; the virtual "
       "member function is not instantiated but it might be with a "
       "different compiler");
  diag(ImplicitSpecialization->getPointOfInstantiation(),
       "template instantiated here", DiagnosticIDs::Note);
}

} // namespace clang::tidy::portability
