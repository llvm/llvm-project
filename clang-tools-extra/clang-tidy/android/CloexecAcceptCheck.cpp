//===--- CloexecAcceptCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  const std::string &ReplacementText =
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
