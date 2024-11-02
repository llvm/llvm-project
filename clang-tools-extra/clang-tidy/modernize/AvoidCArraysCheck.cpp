//===--- AvoidCArraysCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidCArraysCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

AST_MATCHER(clang::TypeLoc, hasValidBeginLoc) {
  return Node.getBeginLoc().isValid();
}

AST_MATCHER_P(clang::TypeLoc, hasType,
              clang::ast_matchers::internal::Matcher<clang::Type>,
              InnerMatcher) {
  const clang::Type *TypeNode = Node.getTypePtr();
  return TypeNode != nullptr &&
         InnerMatcher.matches(*TypeNode, Finder, Builder);
}

AST_MATCHER(clang::RecordDecl, isExternCContext) {
  return Node.isExternCContext();
}

AST_MATCHER(clang::ParmVarDecl, isArgvOfMain) {
  const clang::DeclContext *DC = Node.getDeclContext();
  const auto *FD = llvm::dyn_cast<clang::FunctionDecl>(DC);
  return FD ? FD->isMain() : false;
}

} // namespace

AvoidCArraysCheck::AvoidCArraysCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowStringArrays(Options.get("AllowStringArrays", false)) {}

void AvoidCArraysCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowStringArrays", AllowStringArrays);
}

void AvoidCArraysCheck::registerMatchers(MatchFinder *Finder) {
  ast_matchers::internal::Matcher<TypeLoc> IgnoreStringArrayIfNeededMatcher =
      anything();
  if (AllowStringArrays)
    IgnoreStringArrayIfNeededMatcher =
        unless(typeLoc(loc(hasCanonicalType(incompleteArrayType(
                           hasElementType(isAnyCharacter())))),
                       hasParent(varDecl(hasInitializer(stringLiteral()),
                                         unless(parmVarDecl())))));

  Finder->addMatcher(
      typeLoc(hasValidBeginLoc(), hasType(arrayType()),
              unless(anyOf(hasParent(parmVarDecl(isArgvOfMain())),
                           hasParent(varDecl(isExternC())),
                           hasParent(fieldDecl(
                               hasParent(recordDecl(isExternCContext())))),
                           hasAncestor(functionDecl(isExternC())))),
              std::move(IgnoreStringArrayIfNeededMatcher))
          .bind("typeloc"),
      this);
}

void AvoidCArraysCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ArrayType = Result.Nodes.getNodeAs<TypeLoc>("typeloc");

  diag(ArrayType->getBeginLoc(),
       "do not declare %select{C-style|C VLA}0 arrays, use "
       "%select{std::array<>|std::vector<>}0 instead")
      << ArrayType->getTypePtr()->isVariableArrayType();
}

} // namespace clang::tidy::modernize
