//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MultipleInheritanceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {
AST_MATCHER(CXXRecordDecl, hasBases) {
  return Node.hasDefinition() && Node.getNumBases() > 0;
}
} // namespace

bool MultipleInheritanceCheck::isInterface(const CXXBaseSpecifier &Base) {
  const CXXRecordDecl *const Node = Base.getType()->getAsCXXRecordDecl();
  if (!Node)
    return true;

  assert(Node->isCompleteDefinition());

  // Short circuit the lookup if we have analyzed this record before.
  if (const auto CachedValue = InterfaceMap.find(Node);
      CachedValue != InterfaceMap.end())
    return CachedValue->second;

  // To be an interface, a class must have...
  const bool CurrentClassIsInterface =
      // ...no bases that aren't interfaces...
      llvm::none_of(Node->bases(),
                    [&](const CXXBaseSpecifier &I) {
                      return !I.isVirtual() && !isInterface(I);
                    }) &&
      // ...no fields, and...
      Node->field_empty() &&
      // ...no methods that aren't pure virtual.
      llvm::none_of(Node->methods(), [](const CXXMethodDecl *M) {
        return M->isUserProvided() && !M->isPureVirtual() && !M->isStatic();
      });

  InterfaceMap.try_emplace(Node, CurrentClassIsInterface);
  return CurrentClassIsInterface;
}

void MultipleInheritanceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxRecordDecl(hasBases(), isDefinition()).bind("decl"),
                     this);
}

void MultipleInheritanceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &D = *Result.Nodes.getNodeAs<CXXRecordDecl>("decl");
  // Check to see if the class inherits from multiple concrete classes.
  unsigned NumConcrete =
      llvm::count_if(D.bases(), [&](const CXXBaseSpecifier &I) {
        return !I.isVirtual() && !isInterface(I);
      });

  // Check virtual bases to see if there is more than one concrete
  // non-virtual base.
  NumConcrete += llvm::count_if(
      D.vbases(), [&](const CXXBaseSpecifier &V) { return !isInterface(V); });

  if (NumConcrete > 1)
    diag(D.getBeginLoc(), "inheriting multiple classes that aren't "
                          "pure virtual is discouraged");
}

} // namespace clang::tidy::misc
