//===--- CloexecDupCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecDupCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecDupCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersImpl(Finder,
                       functionDecl(returns(isInteger()), hasName("dup"),
                                    hasParameter(0, hasType(isInteger()))));
}

void CloexecDupCheck::check(const MatchFinder::MatchResult &Result) {
  std::string ReplacementText =
      (Twine("fcntl(") + getSpellingArg(Result, 0) + ", F_DUPFD_CLOEXEC)")
          .str();

  replaceFunc(Result,
              "prefer fcntl() to dup() because fcntl() allows F_DUPFD_CLOEXEC",
              ReplacementText);
}

} // namespace android
} // namespace tidy
} // namespace clang
