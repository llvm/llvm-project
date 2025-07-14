//===--- CloexecAccept4Check.cpp - clang-tidy------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecAccept4Check.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

void CloexecAccept4Check::registerMatchers(MatchFinder *Finder) {
  auto SockAddrPointerType =
      hasType(pointsTo(recordDecl(isStruct(), hasName("sockaddr"))));
  auto SockLenPointerType = hasType(pointsTo(namedDecl(hasName("socklen_t"))));

  registerMatchersImpl(Finder,
                       functionDecl(returns(isInteger()), hasName("accept4"),
                                    hasParameter(0, hasType(isInteger())),
                                    hasParameter(1, SockAddrPointerType),
                                    hasParameter(2, SockLenPointerType),
                                    hasParameter(3, hasType(isInteger()))));
}

void CloexecAccept4Check::check(const MatchFinder::MatchResult &Result) {
  insertMacroFlag(Result, /*MacroFlag=*/"SOCK_CLOEXEC", /*ArgPos=*/3);
}

} // namespace clang::tidy::android
