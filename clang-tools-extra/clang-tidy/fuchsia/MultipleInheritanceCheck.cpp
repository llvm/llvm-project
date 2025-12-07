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

namespace clang::tidy::fuchsia {

namespace {
AST_MATCHER(CXXRecordDecl, hasBases) {
  if (Node.hasDefinition())
    return Node.getNumBases() > 0;
  return false;
}
} // namespace

bool MultipleInheritanceCheck::isInterface(const CXXRecordDecl *Node) {
  if (!Node)
    return false;

  assert(Node->isCompleteDefinition());

  // Short circuit the lookup if we have analyzed this record before.
  const auto CachedValue = InterfaceMap.find(Node);
  if (CachedValue != InterfaceMap.end())
    return CachedValue->second;

  const bool CurrentClassIsInterface = [&] {
    // To be an interface, all base classes must be interfaces as well.
    for (const CXXBaseSpecifier &I : Node->bases()) {
      if (!I.isVirtual() && !isInterface(I.getType()->getAsCXXRecordDecl()))
        return false;
    }

    // Interfaces should have no fields.
    if (!Node->field_empty())
      return false;

    // Interfaces should have exclusively pure virtual methods.
    return llvm::none_of(Node->methods(), [](const CXXMethodDecl *M) {
      return M->isUserProvided() && !M->isPureVirtual() && !M->isStatic();
    });
  }();
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
  unsigned NumConcrete = 0;
  for (const CXXBaseSpecifier &I : D.bases()) {
    if (!I.isVirtual() && !isInterface(I.getType()->getAsCXXRecordDecl()))
      ++NumConcrete;
  }

  // Check virtual bases to see if there is more than one concrete
  // non-virtual base.
  for (const CXXBaseSpecifier &V : D.vbases()) {
    if (!isInterface(V.getType()->getAsCXXRecordDecl()))
      ++NumConcrete;
  }

  if (NumConcrete > 1) {
    diag(D.getBeginLoc(), "inheriting multiple classes that aren't "
                          "pure virtual is discouraged");
  }
}

} // namespace clang::tidy::fuchsia
