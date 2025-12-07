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

bool MultipleInheritanceCheck::isCurrentClassInterface(
    const CXXRecordDecl *Node) const {
  // Interfaces should have no fields.
  if (!Node->field_empty())
    return false;

  // Interfaces should have exclusively pure methods.
  return llvm::none_of(Node->methods(), [](const CXXMethodDecl *M) {
    return M->isUserProvided() && !M->isPureVirtual() && !M->isStatic();
  });
}

bool MultipleInheritanceCheck::isInterface(const CXXRecordDecl *Node) {
  // Short circuit the lookup if we have analyzed this record before.
  const auto CachedValue = InterfaceMap.find(Node);
  if (CachedValue != InterfaceMap.end())
    return CachedValue->second;

  // To be an interface, all base classes must be interfaces as well.
  for (const auto &I : Node->bases()) {
    if (I.isVirtual())
      continue;
    const auto *Base = I.getType()->getAsCXXRecordDecl();
    if (!Base)
      continue;
    assert(Base->isCompleteDefinition());
    if (!isInterface(Base)) {
      InterfaceMap.try_emplace(Node, false);
      return false;
    }
  }

  const bool CurrentClassIsInterface = isCurrentClassInterface(Node);
  InterfaceMap.try_emplace(Node, CurrentClassIsInterface);
  return CurrentClassIsInterface;
}

void MultipleInheritanceCheck::registerMatchers(MatchFinder *Finder) {
  // Match declarations which have bases.
  Finder->addMatcher(cxxRecordDecl(hasBases(), isDefinition()).bind("decl"),
                     this);
}

void MultipleInheritanceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &D = *Result.Nodes.getNodeAs<CXXRecordDecl>("decl");
  // Check against map to see if the class inherits from multiple
  // concrete classes
  unsigned NumConcrete = 0;
  for (const auto &I : D.bases()) {
    if (I.isVirtual())
      continue;
    const auto *Base = I.getType()->getAsCXXRecordDecl();
    if (!Base)
      continue;
    assert(Base->isCompleteDefinition());
    if (!isInterface(Base))
      ++NumConcrete;
  }

  // Check virtual bases to see if there is more than one concrete
  // non-virtual base.
  for (const auto &V : D.vbases()) {
    const auto *Base = V.getType()->getAsCXXRecordDecl();
    if (!Base)
      continue;
    assert(Base->isCompleteDefinition());
    if (!isInterface(Base))
      ++NumConcrete;
  }

  if (NumConcrete > 1) {
    diag(D.getBeginLoc(), "inheriting multiple classes that aren't "
                          "pure virtual is discouraged");
  }
}

} // namespace clang::tidy::fuchsia
