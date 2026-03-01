//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ForbidNonVirtualBaseDtorCheck.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void ForbidNonVirtualBaseDtorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxRecordDecl(
          isDefinition(), has(fieldDecl()),
          hasAnyBase(
              cxxBaseSpecifier(isPublic(),
                               hasType(cxxRecordDecl(
                                   isDefinition(),
                                   unless(has(cxxDestructorDecl(isVirtual()))),
                                   unless(has(cxxDestructorDecl(
                                       isProtected(), unless(isVirtual())))))))
                  .bind("base")))
          .bind("derived"),
      this);
}

void ForbidNonVirtualBaseDtorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Derived = Result.Nodes.getNodeAs<CXXRecordDecl>("derived");
  const auto *Base = Result.Nodes.getNodeAs<CXXBaseSpecifier>("base");
  if (!Derived || !Base)
    return;

  const auto *BaseType = Base->getType()->getAsCXXRecordDecl();
  if (!BaseType)
    return;

  diag(Derived->getLocation(),
       "class '%0' inherits from '%1' which has a non-virtual destructor")
      << Derived->getName() << BaseType->getName();
}

} // namespace clang::tidy::misc
