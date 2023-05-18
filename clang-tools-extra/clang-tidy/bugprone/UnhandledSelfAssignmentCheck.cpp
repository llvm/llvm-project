//===--- UnhandledSelfAssignmentCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnhandledSelfAssignmentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

UnhandledSelfAssignmentCheck::UnhandledSelfAssignmentCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnlyIfThisHasSuspiciousField(
          Options.get("WarnOnlyIfThisHasSuspiciousField", true)) {}

void UnhandledSelfAssignmentCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnlyIfThisHasSuspiciousField",
                WarnOnlyIfThisHasSuspiciousField);
}

void UnhandledSelfAssignmentCheck::registerMatchers(MatchFinder *Finder) {
  // We don't care about deleted, default or implicit operator implementations.
  const auto IsUserDefined = cxxMethodDecl(
      isDefinition(), unless(anyOf(isDeleted(), isImplicit(), isDefaulted())));

  // We don't need to worry when a copy assignment operator gets the other
  // object by value.
  const auto HasReferenceParam =
      cxxMethodDecl(hasParameter(0, parmVarDecl(hasType(referenceType()))));

  // Self-check: Code compares something with 'this' pointer. We don't check
  // whether it is actually the parameter what we compare.
  const auto HasNoSelfCheck = cxxMethodDecl(unless(hasDescendant(
      binaryOperation(hasAnyOperatorName("==", "!="),
                      hasEitherOperand(ignoringParenCasts(cxxThisExpr()))))));

  // Both copy-and-swap and copy-and-move method creates a copy first and
  // assign it to 'this' with swap or move.
  // In the non-template case, we can search for the copy constructor call.
  const auto HasNonTemplateSelfCopy = cxxMethodDecl(
      ofClass(cxxRecordDecl(unless(hasAncestor(classTemplateDecl())))),
      traverse(TK_AsIs,
               hasDescendant(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                   isCopyConstructor(), ofClass(equalsBoundNode("class"))))))));

  // In the template case, we need to handle two separate cases: 1) a local
  // variable is created with the copy, 2) copy is created only as a temporary
  // object.
  const auto HasTemplateSelfCopy = cxxMethodDecl(
      ofClass(cxxRecordDecl(hasAncestor(classTemplateDecl()))),
      anyOf(hasDescendant(
                varDecl(hasType(cxxRecordDecl(equalsBoundNode("class"))),
                        hasDescendant(parenListExpr()))),
            hasDescendant(cxxUnresolvedConstructExpr(hasDescendant(declRefExpr(
                hasType(cxxRecordDecl(equalsBoundNode("class")))))))));

  // If inside the copy assignment operator another assignment operator is
  // called on 'this' we assume that self-check might be handled inside
  // this nested operator.
  const auto HasNoNestedSelfAssign =
      cxxMethodDecl(unless(hasDescendant(cxxMemberCallExpr(callee(cxxMethodDecl(
          hasName("operator="), ofClass(equalsBoundNode("class"))))))));

  DeclarationMatcher AdditionalMatcher = cxxMethodDecl();
  if (WarnOnlyIfThisHasSuspiciousField) {
    // Matcher for standard smart pointers.
    const auto SmartPointerType = qualType(hasUnqualifiedDesugaredType(
        recordType(hasDeclaration(classTemplateSpecializationDecl(
            hasAnyName("::std::shared_ptr", "::std::unique_ptr",
                       "::std::weak_ptr", "::std::auto_ptr"),
            templateArgumentCountIs(1))))));

    // We will warn only if the class has a pointer or a C array field which
    // probably causes a problem during self-assignment (e.g. first resetting
    // the pointer member, then trying to access the object pointed by the
    // pointer, or memcpy overlapping arrays).
    AdditionalMatcher = cxxMethodDecl(ofClass(cxxRecordDecl(
        has(fieldDecl(anyOf(hasType(pointerType()), hasType(SmartPointerType),
                            hasType(arrayType())))))));
  }

  Finder->addMatcher(cxxMethodDecl(ofClass(cxxRecordDecl().bind("class")),
                                   isCopyAssignmentOperator(), IsUserDefined,
                                   HasReferenceParam, HasNoSelfCheck,
                                   unless(HasNonTemplateSelfCopy),
                                   unless(HasTemplateSelfCopy),
                                   HasNoNestedSelfAssign, AdditionalMatcher)
                         .bind("copyAssignmentOperator"),
                     this);
}

void UnhandledSelfAssignmentCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<CXXMethodDecl>("copyAssignmentOperator");
  diag(MatchedDecl->getLocation(),
       "operator=() does not handle self-assignment properly");
}

} // namespace clang::tidy::bugprone
