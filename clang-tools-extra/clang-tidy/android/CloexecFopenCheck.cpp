//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecFopenCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

void CloexecFopenCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));
  registerMatchersImpl(
      Finder, functionDecl(isExternC(), returns(asString("FILE *")),
                           hasName("fopen"), hasParameter(0, CharPointerType),
                           hasParameter(1, CharPointerType)));
}

void CloexecFopenCheck::check(const MatchFinder::MatchResult &Result) {
  insertStringFlag(Result, /*Mode=*/'e', /*ArgPos=*/1);
}

} // namespace clang::tidy::android
