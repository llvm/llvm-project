//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: 2025 Siemens Corporation and/or its affiliates
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MethodHidingCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <stack>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

bool sameBasicType(ParmVarDecl const *Lhs, ParmVarDecl const *Rhs) {
  if (Lhs && Rhs) {
    return Lhs->getType()
               .getCanonicalType()
               .getNonReferenceType()
               .getUnqualifiedType() == Rhs->getType()
                                            .getCanonicalType()
                                            .getNonReferenceType()
                                            .getUnqualifiedType();
  }
  return false;
}

bool namesCollide(CXXMethodDecl const &Lhs, CXXMethodDecl const &Rhs) {
  if (Lhs.getNameAsString() != Rhs.getNameAsString()) {
    return false;
  }
  if (Lhs.isConst() != Rhs.isConst()) {
    return false;
  }
  if (Lhs.getNumParams() != Rhs.getNumParams()) {
    return false;
  }
  for (unsigned int It = 0; It < Lhs.getNumParams(); ++It) {
    if (!sameBasicType(Lhs.getParamDecl(It), Rhs.getParamDecl(It))) {
      return false;
    }
  }
  // Templates are not handled yet
  if (Lhs.isTemplated() || Rhs.isTemplated()) {
    return false;
  }
  if (Lhs.isTemplateInstantiation() || Rhs.isTemplateInstantiation()) {
    return false;
  }
  if (Lhs.isFunctionTemplateSpecialization() ||
      Rhs.isFunctionTemplateSpecialization()) {
    return false;
  }
  return true;
}

AST_MATCHER(CXXMethodDecl, nameCollidesWithMethodInBase) {
  CXXRecordDecl const *DerivedClass = Node.getParent();
  for (auto const &Base : DerivedClass->bases()) {
    std::stack<CXXBaseSpecifier const *> Stack;
    Stack.push(&Base);
    while (!Stack.empty()) {
      CXXBaseSpecifier const *CurrentBaseSpec = Stack.top();
      Stack.pop();

      if (CurrentBaseSpec->getAccessSpecifier() ==
          clang::AccessSpecifier::AS_private) {
        continue;
      }

      const auto *CurrentRecord =
          CurrentBaseSpec->getType()->getAsCXXRecordDecl();
      if (!CurrentRecord) {
        continue;
      }

      for (auto const &BaseMethod : CurrentRecord->methods()) {
        if (namesCollide(*BaseMethod, Node)) {
          ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
          Builder->setBinding("base_method",
                              clang::DynTypedNode::create(*BaseMethod));
          return true;
        }
      }

      for (auto const &SubBase : CurrentRecord->bases()) {
        Stack.push(&SubBase);
      }
    }
  }
  return false;
}

// Same as clang-tools-extra/clang-tidy/modernize/UseEqualsDefaultCheck.cpp,
// similar matchers are used elsewhere in LLVM
AST_MATCHER(CXXMethodDecl, isOutOfLine) { return Node.isOutOfLine(); }

} // namespace

MethodHidingCheck::MethodHidingCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void MethodHidingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(
          unless(anyOf(isOutOfLine(), isStaticStorageClass(), isImplicit(),
                       cxxConstructorDecl(), isOverride(),
                       // isFinal(), //included with isOverride,
                       isPrivate())),
          ofClass(cxxRecordDecl(
                      isDerivedFrom(cxxRecordDecl(unless(isInStdNamespace()))))
                      .bind("derived_class")),
          nameCollidesWithMethodInBase())
          .bind("shadowing_method"),
      this);
}

void MethodHidingCheck::check(const MatchFinder::MatchResult &Result) {
  auto const *ShadowingMethod =
      Result.Nodes.getNodeAs<CXXMethodDecl>("shadowing_method");
  auto const *DerivedClass =
      Result.Nodes.getNodeAs<CXXRecordDecl>("derived_class");
  auto const *BaseMethod = Result.Nodes.getNodeAs<CXXMethodDecl>("base_method");

  if (!ShadowingMethod || !DerivedClass || !BaseMethod) {
    llvm_unreachable("Required binding not found");
  }

  auto const MethodName = ShadowingMethod->getNameInfo().getAsString();
  auto const DerivedClassName = DerivedClass->getNameAsString();
  std::string BaseClassName = BaseMethod->getParent()->getNameAsString();

  std::string Message;
  llvm::raw_string_ostream StringStream(Message);
  StringStream << "'" << ShadowingMethod->getQualifiedNameAsString()
               << "' hides same method in '" << BaseClassName << "'";
  diag(ShadowingMethod->getBeginLoc(), Message);
  diag(BaseMethod->getBeginLoc(), "previous definition is here",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
