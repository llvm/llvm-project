//===--- CloexecInotifyInit1Check.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecInotifyInit1Check.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecInotifyInit1Check::registerMatchers(MatchFinder *Finder) {
  registerMatchersImpl(
      Finder, functionDecl(returns(isInteger()), hasName("inotify_init1"),
                           hasParameter(0, hasType(isInteger()))));
}

void CloexecInotifyInit1Check::check(const MatchFinder::MatchResult &Result) {
  insertMacroFlag(Result, /*MacroFlag=*/"IN_CLOEXEC", /*ArgPos=*/0);
}

} // namespace android
} // namespace tidy
} // namespace clang
