//===--- CloexecPipeCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecPipeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

void CloexecPipeCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersImpl(Finder,
                       functionDecl(returns(isInteger()), hasName("pipe"),
                                    hasParameter(0, hasType(pointsTo(isInteger())))));
}

void CloexecPipeCheck::check(const MatchFinder::MatchResult &Result) {
  std::string ReplacementText =
      (Twine("pipe2(") + getSpellingArg(Result, 0) + ", O_CLOEXEC)").str();

  replaceFunc(
      Result,
      "prefer pipe2() with O_CLOEXEC to avoid leaking file descriptors to child processes",
      ReplacementText);
}

} // namespace clang::tidy::android
