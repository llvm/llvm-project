//===--- CloexecDupCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  const std::string &ReplacementText =
      (Twine("fcntl(") + getSpellingArg(Result, 0) + ", F_DUPFD_CLOEXEC)")
          .str();

  replaceFunc(Result,
              "prefer fcntl() to dup() because fcntl() allows F_DUPFD_CLOEXEC",
              ReplacementText);
}

} // namespace android
} // namespace tidy
} // namespace clang
