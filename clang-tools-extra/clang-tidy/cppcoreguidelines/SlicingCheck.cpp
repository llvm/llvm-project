//===--- SlicingCheck.cpp - clang-tidy-------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SlicingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

void SlicingCheck::registerMatchers(MatchFinder *Finder) {
  // When we see:
  //   class B : public A { ... };
  //   A a;
  //   B b;
  //   a = b;
  // The assignment is OK if:
  //   - the assignment operator is defined as taking a B as second parameter,
  //   or
  //   - B does not define any additional members (either variables or
  //   overrides) wrt A.
  //
  // The same holds for copy ctor calls. This also captures stuff like:
  //   void f(A a);
  //   f(b);

  //  Helpers.
  const auto OfBaseClass = ofClass(cxxRecordDecl().bind("BaseDecl"));
  const auto IsDerivedFromBaseDecl =
      cxxRecordDecl(isDerivedFrom(equalsBoundNode("BaseDecl")))
          .bind("DerivedDecl");
  const auto HasTypeDerivedFromBaseDecl =
      anyOf(hasType(IsDerivedFromBaseDecl),
            hasType(references(IsDerivedFromBaseDecl)));
  const auto IsCallToBaseClass = hasParent(cxxConstructorDecl(
      ofClass(isSameOrDerivedFrom(equalsBoundNode("DerivedDecl"))),
      hasAnyConstructorInitializer(allOf(
          isBaseInitializer(), withInitializer(equalsBoundNode("Call"))))));

  // Assignment slicing: "a = b;" and "a = std::move(b);" variants.
  const auto SlicesObjectInAssignment =
      callExpr(expr().bind("Call"),
               callee(cxxMethodDecl(anyOf(isCopyAssignmentOperator(),
                                          isMoveAssignmentOperator()),
                                    OfBaseClass)),
               hasArgument(1, HasTypeDerivedFromBaseDecl));

  // Construction slicing: "A a{b};" and "f(b);" variants. Note that in case of
  // slicing the letter will create a temporary and therefore call a ctor.
  const auto SlicesObjectInCtor = cxxConstructExpr(
      expr().bind("Call"),
      hasDeclaration(cxxConstructorDecl(
          anyOf(isCopyConstructor(), isMoveConstructor()), OfBaseClass)),
      hasArgument(0, HasTypeDerivedFromBaseDecl),
      // We need to disable matching on the call to the base copy/move
      // constructor in DerivedDecl's constructors.
      unless(IsCallToBaseClass));

  Finder->addMatcher(
      traverse(TK_AsIs, expr(SlicesObjectInAssignment).bind("Call")), this);
  Finder->addMatcher(traverse(TK_AsIs, SlicesObjectInCtor), this);
}

/// Warns on methods overridden in DerivedDecl with respect to BaseDecl.
/// FIXME: this warns on all overrides outside of the sliced path in case of
/// multiple inheritance.
void SlicingCheck::diagnoseSlicedOverriddenMethods(
    const Expr &Call, const CXXRecordDecl &DerivedDecl,
    const CXXRecordDecl &BaseDecl) {
  if (DerivedDecl.getCanonicalDecl() == BaseDecl.getCanonicalDecl())
    return;
  for (const auto *Method : DerivedDecl.methods()) {
    // Virtual destructors are OK. We're ignoring constructors since they are
    // tagged as overrides.
    if (isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method))
      continue;
    if (Method->size_overridden_methods() > 0) {
      diag(Call.getExprLoc(),
           "slicing object from type %0 to %1 discards override %2")
          << &DerivedDecl << &BaseDecl << Method;
    }
  }
  // Recursively process bases.
  for (const auto &Base : DerivedDecl.bases()) {
    if (const auto *BaseRecord = Base.getType()->getAsCXXRecordDecl()) {
      if (BaseRecord->isCompleteDefinition())
        diagnoseSlicedOverriddenMethods(Call, *BaseRecord, BaseDecl);
    }
  }
}

void SlicingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BaseDecl = Result.Nodes.getNodeAs<CXXRecordDecl>("BaseDecl");
  const auto *DerivedDecl =
      Result.Nodes.getNodeAs<CXXRecordDecl>("DerivedDecl");
  const auto *Call = Result.Nodes.getNodeAs<Expr>("Call");
  assert(BaseDecl != nullptr);
  assert(DerivedDecl != nullptr);
  assert(Call != nullptr);

  // Warn when slicing the vtable.
  // We're looking through all the methods in the derived class and see if they
  // override some methods in the base class.
  // It's not enough to just test whether the class is polymorphic because we
  // would be fine slicing B to A if no method in B (or its bases) overrides
  // anything in A:
  //   class A { virtual void f(); };
  //   class B : public A {};
  // because in that case calling A::f is the same as calling B::f.
  diagnoseSlicedOverriddenMethods(*Call, *DerivedDecl, *BaseDecl);

  // Warn when slicing member variables.
  const auto &BaseLayout =
      BaseDecl->getASTContext().getASTRecordLayout(BaseDecl);
  const auto &DerivedLayout =
      DerivedDecl->getASTContext().getASTRecordLayout(DerivedDecl);
  const CharUnits StateSize =
      DerivedLayout.getDataSize() - BaseLayout.getDataSize();
  if (StateSize.isPositive()) {
    diag(Call->getExprLoc(), "slicing object from type %0 to %1 discards "
                             "%2 bytes of state")
        << DerivedDecl << BaseDecl << static_cast<int>(StateSize.getQuantity());
  }
}

} // namespace clang::tidy::cppcoreguidelines
