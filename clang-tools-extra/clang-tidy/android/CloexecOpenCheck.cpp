//===--- CloexecOpenCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecOpenCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

void CloexecOpenCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));
  registerMatchersImpl(Finder,
                       functionDecl(isExternC(), returns(isInteger()),
                                    hasAnyName("open", "open64"),
                                    hasParameter(0, CharPointerType),
                                    hasParameter(1, hasType(isInteger()))));
  registerMatchersImpl(Finder,
                       functionDecl(isExternC(), returns(isInteger()),
                                    hasName("openat"),
                                    hasParameter(0, hasType(isInteger())),
                                    hasParameter(1, CharPointerType),
                                    hasParameter(2, hasType(isInteger()))));
}

void CloexecOpenCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>(FuncDeclBindingStr);
  assert(FD->param_size() > 1);
  int ArgPos = (FD->param_size() > 2) ? 2 : 1;
  insertMacroFlag(Result, /*MacroFlag=*/"O_CLOEXEC", ArgPos);
}

} // namespace clang::tidy::android
