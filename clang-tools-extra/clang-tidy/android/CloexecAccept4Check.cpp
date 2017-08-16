//===--- CloexecAccept4Check.cpp - clang-tidy------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecAccept4Check.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

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
  insertMacroFlag(Result, /*MarcoFlag=*/"SOCK_CLOEXEC", /*ArgPos=*/3);
}

} // namespace android
} // namespace tidy
} // namespace clang
