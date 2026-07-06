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
  // Collect the direct and virtual concrete bases of the class.
  SmallVector<const CXXRecordDecl *> DirectConcreteBases;
  for (const CXXBaseSpecifier &Base : D.bases())
    if (!Base.isVirtual() && !isInterface(Base))
      DirectConcreteBases.push_back(Base.getType()->getAsCXXRecordDecl());

  SmallVector<const CXXRecordDecl *> VirtualConcreteBases;
  for (const CXXBaseSpecifier &VBase : D.vbases())
    if (!isInterface(VBase))
      VirtualConcreteBases.push_back(VBase.getType()->getAsCXXRecordDecl());

  unsigned NumConcrete = DirectConcreteBases.size();

  // Count only virtual concrete bases that introduce an additional
  // implementation base, skipping those already represented by a more derived
  // concrete base.
  NumConcrete += llvm::count_if(
      VirtualConcreteBases, [&](const CXXRecordDecl *VirtualBase) {
        const bool HiddenByMoreDerivedVirtualBase = llvm::any_of(
            VirtualConcreteBases, [&](const CXXRecordDecl *OtherVirtualBase) {
              return VirtualBase != OtherVirtualBase &&
                     OtherVirtualBase->isVirtuallyDerivedFrom(VirtualBase);
            });
        const bool HiddenByDirectConcreteBase = llvm::any_of(
            DirectConcreteBases, [&](const CXXRecordDecl *DirectBase) {
              return DirectBase->isVirtuallyDerivedFrom(VirtualBase);
            });
        return !HiddenByMoreDerivedVirtualBase && !HiddenByDirectConcreteBase;
      });

  if (NumConcrete > 1)
    diag(D.getBeginLoc(), "inheriting multiple classes that aren't "
                          "pure virtual is discouraged");
}

} // namespace clang::tidy::misc
