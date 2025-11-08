//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DerivedMethodShadowingBaseMethodCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static bool sameBasicType(const ParmVarDecl *Lhs, const ParmVarDecl *Rhs) {
  return Lhs && Rhs &&
         Lhs->getType()
                 .getCanonicalType()
                 .getNonReferenceType()
                 .getUnqualifiedType() == Rhs->getType()
                                              .getCanonicalType()
                                              .getNonReferenceType()
                                              .getUnqualifiedType();
}

static bool namesCollide(const CXXMethodDecl &Lhs, const CXXMethodDecl &Rhs) {
  if (Lhs.getNameAsString() != Rhs.getNameAsString())
    return false;
  if (Lhs.isConst() != Rhs.isConst())
    return false;
  if (Lhs.getNumParams() != Rhs.getNumParams())
    return false;
  for (unsigned int It = 0; It < Lhs.getNumParams(); ++It)
    if (!sameBasicType(Lhs.getParamDecl(It), Rhs.getParamDecl(It)))
      return false;
  return true;
}

namespace {

AST_MATCHER(CXXMethodDecl, nameCollidesWithMethodInBase) {
  const CXXRecordDecl *DerivedClass = Node.getParent();
  for (const auto &Base : DerivedClass->bases()) {
    llvm::SmallVector<const CXXBaseSpecifier *, 8> Stack;
    Stack.push_back(&Base);
    while (!Stack.empty()) {
      const CXXBaseSpecifier *CurrentBaseSpec = Stack.back();
      Stack.pop_back();

      if (CurrentBaseSpec->getAccessSpecifier() ==
          clang::AccessSpecifier::AS_private)
        continue;

      const CXXRecordDecl *CurrentRecord =
          CurrentBaseSpec->getType()->getAsCXXRecordDecl();
      if (!CurrentRecord)
        continue;

      // For multiple inheritance, we ignore only the bases that come from the
      // std:: namespace
      if (CurrentRecord->isInStdNamespace())
        continue;

      for (const auto &BaseMethod : CurrentRecord->methods()) {
        if (namesCollide(*BaseMethod, Node)) {
          const ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
          Builder->setBinding("base_method",
                              clang::DynTypedNode::create(*BaseMethod));
          return true;
        }
      }

      for (const auto &SubBase : CurrentRecord->bases())
        Stack.push_back(&SubBase);
    }
  }
  return false;
}

// Same as clang-tools-extra/clang-tidy/modernize/UseEqualsDefaultCheck.cpp,
// similar matchers are used elsewhere in LLVM
AST_MATCHER(CXXMethodDecl, isOutOfLine) { return Node.isOutOfLine(); }

} // namespace

DerivedMethodShadowingBaseMethodCheck::DerivedMethodShadowingBaseMethodCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void DerivedMethodShadowingBaseMethodCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(
          unless(anyOf(isOutOfLine(), isStaticStorageClass(), isImplicit(),
                       cxxConstructorDecl(), isOverride(), isPrivate(),
                       // isFinal(), //included with isOverride,
                       // Templates are not handled yet
                       ast_matchers::isTemplateInstantiation(),
                       ast_matchers::isExplicitTemplateSpecialization())),
          ofClass(cxxRecordDecl(isDerivedFrom(cxxRecordDecl()))
                      .bind("derived_class")),
          nameCollidesWithMethodInBase())
          .bind("shadowing_method"),
      this);
}

void DerivedMethodShadowingBaseMethodCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *ShadowingMethod =
      Result.Nodes.getNodeAs<CXXMethodDecl>("shadowing_method");
  const auto *DerivedClass =
      Result.Nodes.getNodeAs<CXXRecordDecl>("derived_class");
  const auto *BaseMethod = Result.Nodes.getNodeAs<CXXMethodDecl>("base_method");

  if (!ShadowingMethod || !DerivedClass || !BaseMethod)
    llvm_unreachable("Required binding not found");

  diag(ShadowingMethod->getBeginLoc(),
       "'%0' shadows method with the same name in class %1")
      << ShadowingMethod->getQualifiedNameAsString() << BaseMethod->getParent();
  diag(BaseMethod->getBeginLoc(), "previous definition of %0 is here",
       DiagnosticIDs::Note)
      << ShadowingMethod;
}

} // namespace clang::tidy::bugprone
