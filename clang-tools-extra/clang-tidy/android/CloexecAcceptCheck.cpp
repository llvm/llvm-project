//===--- CloexecAcceptCheck.cpp - clang-tidy-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecAcceptCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecAcceptCheck::registerMatchers(MatchFinder *Finder) {
  auto SockAddrPointerType =
      hasType(pointsTo(recordDecl(isStruct(), hasName("sockaddr"))));
  auto SockLenPointerType = hasType(pointsTo(namedDecl(hasName("socklen_t"))));

  registerMatchersImpl(Finder,
                       functionDecl(returns(isInteger()), hasName("accept"),
                                    hasParameter(0, hasType(isInteger())),
                                    hasParameter(1, SockAddrPointerType),
                                    hasParameter(2, SockLenPointerType)));
}

void CloexecAcceptCheck::check(const MatchFinder::MatchResult &Result) {
  std::string ReplacementText =
      (Twine("accept4(") + getSpellingArg(Result, 0) + ", " +
       getSpellingArg(Result, 1) + ", " + getSpellingArg(Result, 2) +
       ", SOCK_CLOEXEC)")
          .str();

  replaceFunc(
      Result,
      "prefer accept4() to accept() because accept4() allows SOCK_CLOEXEC",
      ReplacementText);
}

} // namespace android
} // namespace tidy
} // namespace clang
