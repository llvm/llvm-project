//===--- SuspiciousPointerArithmeticsUsingSizeofCheck.cpp - clang-tidy --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousPointerArithmeticsUsingSizeofCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace {
  static const char *bin_op_bind = "ptr-sizeof-expression";	
}
namespace clang::tidy::bugprone {

SuspiciousPointerArithmeticsUsingSizeofCheck::SuspiciousPointerArithmeticsUsingSizeofCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(
	     expr(anyOf(
                    binaryOperator(hasAnyOperatorName("+","-"),
                      hasEitherOperand(hasType(pointerType())),
		      hasEitherOperand(sizeOfExpr(expr())),
		      unless(allOf(hasLHS(hasType(pointerType())),
				   hasRHS(hasType(pointerType()))))
		      ).bind(bin_op_bind),
		    binaryOperator(hasAnyOperatorName("+=","-="),
	              hasLHS(hasType(pointerType())),
		      hasRHS(sizeOfExpr(expr()))
		      ).bind(bin_op_bind)
            )),
        this);
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::check(const MatchFinder::MatchResult &Result) {
    static const char *diag_msg	= "Suspicious pointer arithmetics using sizeof() operator";
    auto Matched = Result.Nodes.getNodeAs<BinaryOperator>(bin_op_bind);
    diag(Matched->getExprLoc(),diag_msg)<< Matched->getSourceRange();
}

} // namespace clang::tidy::bugprone
