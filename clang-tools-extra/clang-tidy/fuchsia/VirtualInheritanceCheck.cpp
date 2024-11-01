//===--- VirtualInheritanceCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VirtualInheritanceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::fuchsia {

namespace {
AST_MATCHER(CXXRecordDecl, hasDirectVirtualBaseClass) {
  if (!Node.hasDefinition()) return false;
  if (!Node.getNumVBases()) return false;
  for (const CXXBaseSpecifier &Base : Node.bases())
    if (Base.isVirtual()) return true;
  return false;
}
} // namespace

void VirtualInheritanceCheck::registerMatchers(MatchFinder *Finder) {
  // Defining classes using direct virtual inheritance is disallowed.
  Finder->addMatcher(cxxRecordDecl(hasDirectVirtualBaseClass()).bind("decl"),
                     this);
}

void VirtualInheritanceCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<CXXRecordDecl>("decl"))
    diag(D->getBeginLoc(), "direct virtual inheritance is disallowed");
}

} // namespace clang::tidy::fuchsia
