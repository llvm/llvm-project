//===--- CloexecEpollCreateCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecEpollCreateCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecEpollCreateCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersImpl(
      Finder, functionDecl(returns(isInteger()), hasName("epoll_create"),
                           hasParameter(0, hasType(isInteger()))));
}

void CloexecEpollCreateCheck::check(const MatchFinder::MatchResult &Result) {
  replaceFunc(Result,
              "prefer epoll_create() to epoll_create1() "
              "because epoll_create1() allows "
              "EPOLL_CLOEXEC",
              "epoll_create1(EPOLL_CLOEXEC)");
}

} // namespace android
} // namespace tidy
} // namespace clang
