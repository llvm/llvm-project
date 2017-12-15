//===--- VirtualInheritanceCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "VirtualInheritanceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

AST_MATCHER(CXXRecordDecl, hasDirectVirtualBaseClass) {
  if (!Node.hasDefinition()) return false;
  if (!Node.getNumVBases()) return false;
  for (const CXXBaseSpecifier &Base : Node.bases())
    if (Base.isVirtual()) return true;
  return false;
}

void VirtualInheritanceCheck::registerMatchers(MatchFinder *Finder) {
  // Defining classes using direct virtual inheritance is disallowed.
  Finder->addMatcher(cxxRecordDecl(hasDirectVirtualBaseClass()).bind("decl"),
                     this);
}

void VirtualInheritanceCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<CXXRecordDecl>("decl"))
    diag(D->getLocStart(), "direct virtual inheritance is disallowed");
}

}  // namespace fuchsia
}  // namespace tidy
}  // namespace clang
