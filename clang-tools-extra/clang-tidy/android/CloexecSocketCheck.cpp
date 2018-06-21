//===--- CloexecSocketCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecSocketCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecSocketCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersImpl(Finder,
                       functionDecl(isExternC(), returns(isInteger()),
                                    hasName("socket"),
                                    hasParameter(0, hasType(isInteger())),
                                    hasParameter(1, hasType(isInteger())),
                                    hasParameter(2, hasType(isInteger()))));
}

void CloexecSocketCheck::check(const MatchFinder::MatchResult &Result) {
  insertMacroFlag(Result, /*MacroFlag=*/"SOCK_CLOEXEC", /*ArgPos=*/1);
}

} // namespace android
} // namespace tidy
} // namespace clang
