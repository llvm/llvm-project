//===--- AvoidCArraysCheck.cpp - clang-tidy -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidCArraysCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

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

namespace clang {
namespace tidy {
namespace modernize {

void AvoidCArraysCheck::registerMatchers(MatchFinder *Finder) {
  // std::array<> is avaliable since C++11.
  if (!getLangOpts().CPlusPlus11)
    return;

  Finder->addMatcher(
      typeLoc(hasValidBeginLoc(), hasType(arrayType()),
              unless(anyOf(hasParent(parmVarDecl(isArgvOfMain())),
                           hasParent(varDecl(isExternC())),
                           hasParent(fieldDecl(
                               hasParent(recordDecl(isExternCContext())))),
                           hasAncestor(functionDecl(isExternC())))))
          .bind("typeloc"),
      this);
}

void AvoidCArraysCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ArrayType = Result.Nodes.getNodeAs<TypeLoc>("typeloc");

  static constexpr llvm::StringLiteral UseArray = llvm::StringLiteral(
      "do not declare C-style arrays, use std::array<> instead");
  static constexpr llvm::StringLiteral UseVector = llvm::StringLiteral(
      "do not declare C VLA arrays, use std::vector<> instead");

  diag(ArrayType->getBeginLoc(),
       ArrayType->getTypePtr()->isVariableArrayType() ? UseVector : UseArray);
}

} // namespace modernize
} // namespace tidy
} // namespace clang
