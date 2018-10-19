//===--- NonPrivateMemberVariablesInClassesCheck.cpp - clang-tidy ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NonPrivateMemberVariablesInClassesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {

AST_MATCHER(CXXRecordDecl, hasMethods) {
  return std::distance(Node.method_begin(), Node.method_end()) != 0;
}

AST_MATCHER(CXXRecordDecl, hasNonStaticMethod) {
  return hasMethod(unless(isStaticStorageClass()))
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

void NonPrivateMemberVariablesInClassesCheck::registerMatchers(
    MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // We can ignore structs/classes with all member variables being public.
  auto ShouldIgnoreRecord =
      allOf(boolean(IgnoreClassesWithAllMemberVariablesBeingPublic),
            unless(hasNonPublicMemberVariable()));

  // We only want the records that not only contain the mutable data (non-static
  // member variables), but also have some logic (non-static member functions).
  // We may optionally ignore records where all the member variables are public.
  auto RecordIsInteresting =
      allOf(anyOf(isStruct(), isClass()), hasMethods(), hasNonStaticMethod(),
            unless(ShouldIgnoreRecord));

  // There are three visibility types: public, protected, private.
  // If we are ok with public fields, then we only want to complain about
  // protected fields, else we want to complain about all non-private fields.
  // We can ignore public member variables in structs/classes, in unions.
  auto InterestingField = fieldDecl(
      IgnorePublicMemberVariables ? isProtected() : unless(isPrivate()));

  Finder->addMatcher(cxxRecordDecl(RecordIsInteresting,
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

} // namespace misc
} // namespace tidy
} // namespace clang
