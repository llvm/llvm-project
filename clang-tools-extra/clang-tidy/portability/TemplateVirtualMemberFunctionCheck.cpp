//===--- TemplateVirtualMemberFunctionCheck.cpp - clang-tidy
//-------------------===//
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

void TemplateVirtualMemberFunctionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(classTemplateSpecializationDecl().bind("specialization"),
                     this);
}

void TemplateVirtualMemberFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("specialization");

  if (MatchedDecl->isExplicitSpecialization())
    return;

  for (auto &&Method : MatchedDecl->methods()) {
    if (!Method->isVirtual())
      continue;

    if (const auto *Dtor = llvm::dyn_cast<CXXDestructorDecl>(Method);
        Dtor && Dtor->isDefaulted())
      continue;

    if (!Method->isUsed()) {
      diag(Method->getLocation(),
           "unspecified virtual member function instantiation; the virtual "
           "member function is not instantiated but it might be with a "
           "different compiler");
      diag(MatchedDecl->getPointOfInstantiation(), "template instantiated here",
           DiagnosticIDs::Note);
    }
  }
}

} // namespace clang::tidy::portability
