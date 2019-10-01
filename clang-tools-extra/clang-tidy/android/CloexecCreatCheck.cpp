//===--- CloexecCreatCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecCreatCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecCreatCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));
  auto MODETType = hasType(namedDecl(hasName("mode_t")));
  registerMatchersImpl(Finder,
                       functionDecl(isExternC(), returns(isInteger()),
                                    hasName("creat"),
                                    hasParameter(0, CharPointerType),
                                    hasParameter(1, MODETType)));
}

void CloexecCreatCheck::check(const MatchFinder::MatchResult &Result) {
  const std::string &ReplacementText =
      (Twine("open (") + getSpellingArg(Result, 0) +
       ", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, " +
       getSpellingArg(Result, 1) + ")")
          .str();
  replaceFunc(Result,
              "prefer open() to creat() because open() allows O_CLOEXEC",
              ReplacementText);
}

} // namespace android
} // namespace tidy
} // namespace clang
