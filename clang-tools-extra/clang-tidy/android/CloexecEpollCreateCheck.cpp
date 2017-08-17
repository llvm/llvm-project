//===--- CloexecEpollCreateCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
