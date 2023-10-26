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

namespace clang::tidy::bugprone {

SuspiciousPointerArithmeticsUsingSizeofCheck::SuspiciousPointerArithmeticsUsingSizeofCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(
		   sizeOfExpr(expr())
            .bind("sizeof-expression"),
        this);
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::check(
    const MatchFinder::MatchResult &Result) {
}

} // namespace clang::tidy::bugprone
