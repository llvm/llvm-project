//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecMemfdCreateCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

void CloexecMemfdCreateCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));
  registerMatchersImpl(
      Finder, functionDecl(returns(isInteger()), hasName("memfd_create"),
                           hasParameter(0, CharPointerType),
                           hasParameter(1, hasType(isInteger()))));
}

void CloexecMemfdCreateCheck::check(const MatchFinder::MatchResult &Result) {
  insertMacroFlag(Result, "MFD_CLOEXEC", /*ArgPos=*/1);
}

} // namespace clang::tidy::android
