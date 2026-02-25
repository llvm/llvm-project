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
#include "clang/Basic/Specifiers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void ForbidNonVirtualBaseDtorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxRecordDecl(isDefinition(),
                    hasAnyBase(cxxBaseSpecifier().bind("BaseSpecifier")))
          .bind("derived"),
      this);
}

void ForbidNonVirtualBaseDtorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Derived = Result.Nodes.getNodeAs<CXXRecordDecl>("derived");
  const auto *BaseSpecifier =
      Result.Nodes.getNodeAs<CXXBaseSpecifier>("BaseSpecifier");
  if (!Derived || !BaseSpecifier)
    return;
  if (BaseSpecifier->getAccessSpecifier() != AS_public)
    return;
  const auto *BaseType = BaseSpecifier->getType()->getAsCXXRecordDecl();
  if (!BaseType || !BaseType->hasDefinition())
    return;
  const auto *Dtor = BaseType->getDestructor();
  if (Dtor && Dtor->isVirtual())
    return;
  if (Dtor && Dtor->getAccess() == AS_protected && !Dtor->isVirtual())
    return;
  if (Derived->isEmpty())
    return;

  diag(Derived->getLocation(),
       "class '%0' inherits from '%1' which has a non-virtual destructor")
      << Derived->getName() << BaseType->getName();
}

} // namespace clang::tidy::misc
