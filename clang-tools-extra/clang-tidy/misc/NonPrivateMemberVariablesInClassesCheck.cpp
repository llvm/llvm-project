//===--- NonPrivateMemberVariablesInClassesCheck.cpp - clang-tidy ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NonPrivateMemberVariablesInClassesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

AST_MATCHER(CXXRecordDecl, hasMethods) {
  return std::distance(Node.method_begin(), Node.method_end()) != 0;
}

AST_MATCHER(CXXRecordDecl, hasNonStaticNonImplicitMethod) {
  return hasMethod(unless(anyOf(isStaticStorageClass(), isImplicit())))
      .matches(Node, Finder, Builder);
}

AST_MATCHER(CXXRecordDecl, hasNonPublicMemberVariable) {
  return cxxRecordDecl(has(fieldDecl(unless(isPublic()))))
      .matches(Node, Finder, Builder);
}

AST_POLYMORPHIC_MATCHER_P(boolean, AST_POLYMORPHIC_SUPPORTED_TYPES(Stmt, Decl),
                          bool, Boolean) {
  return Boolean;
}

} // namespace

NonPrivateMemberVariablesInClassesCheck::
    NonPrivateMemberVariablesInClassesCheck(StringRef Name,
                                            ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreClassesWithAllMemberVariablesBeingPublic(
          Options.get("IgnoreClassesWithAllMemberVariablesBeingPublic", false)),
      IgnorePublicMemberVariables(
          Options.get("IgnorePublicMemberVariables", false)) {}

void NonPrivateMemberVariablesInClassesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreClassesWithAllMemberVariablesBeingPublic",
                IgnoreClassesWithAllMemberVariablesBeingPublic);
  Options.store(Opts, "IgnorePublicMemberVariables",
                IgnorePublicMemberVariables);
}

void NonPrivateMemberVariablesInClassesCheck::registerMatchers(
    MatchFinder *Finder) {
  // We can ignore structs/classes with all member variables being public.
  auto ShouldIgnoreRecord =
      allOf(boolean(IgnoreClassesWithAllMemberVariablesBeingPublic),
            unless(hasNonPublicMemberVariable()));

  // There are three visibility types: public, protected, private.
  // If we are ok with public fields, then we only want to complain about
  // protected fields, else we want to complain about all non-private fields.
  // We can ignore public member variables in structs/classes, in unions.
  auto InterestingField = IgnorePublicMemberVariables
                              ? fieldDecl(isProtected())
                              : fieldDecl(unless(isPrivate()));

  // We only want the records that not only contain the mutable data (non-static
  // member variables), but also have some logic (non-static, non-implicit
  // member functions).  We may optionally ignore records where all the member
  // variables are public.
  Finder->addMatcher(cxxRecordDecl(anyOf(isStruct(), isClass()), hasMethods(),
                                   hasNonStaticNonImplicitMethod(),
                                   unless(ShouldIgnoreRecord),
                                   forEach(InterestingField.bind("field")))
                         .bind("record"),
                     this);
}

void NonPrivateMemberVariablesInClassesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("field");
  assert(Field && "We should have the field we are going to complain about");

  diag(Field->getLocation(), "member variable %0 has %1 visibility")
      << Field << Field->getAccess();
}

} // namespace clang::tidy::misc
