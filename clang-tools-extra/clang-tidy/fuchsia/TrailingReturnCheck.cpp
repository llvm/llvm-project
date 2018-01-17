//===--- TrailingReturnCheck.cpp - clang-tidy------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TrailingReturnCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"

using namespace clang::ast_matchers;

namespace clang {
namespace ast_matchers {

const internal::VariadicDynCastAllOfMatcher<Type, DecltypeType> decltypeType;

} // namespace ast_matchers

namespace tidy {
namespace fuchsia {

AST_MATCHER(FunctionDecl, hasTrailingReturn) {
  return Node.getType()->castAs<FunctionProtoType>()->hasTrailingReturn();
}

void TrailingReturnCheck::registerMatchers(MatchFinder *Finder) {

  // Requires C++11 or later.
  if (!getLangOpts().CPlusPlus11)
    return;

  // Functions that have trailing returns are disallowed, except for those
  // using decltype specifiers and lambda with otherwise unutterable
  // return types.
  Finder->addMatcher(
      functionDecl(allOf(hasTrailingReturn(),
                         unless(anyOf(returns(decltypeType()),
                                      hasParent(cxxRecordDecl(isLambda()))))))
          .bind("decl"),
      this);
}

void TrailingReturnCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<Decl>("decl"))
    diag(D->getLocStart(),
         "a trailing return type is disallowed for this type of declaration");
}

} // namespace fuchsia
} // namespace tidy
} // namespace clang
