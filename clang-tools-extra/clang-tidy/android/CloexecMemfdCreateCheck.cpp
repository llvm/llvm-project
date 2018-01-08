//===--- CloexecMemfdCreateCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecMemfdCreateCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

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

} // namespace android
} // namespace tidy
} // namespace clang
