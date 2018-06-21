//===--- CloexecOpenCheck.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecOpenCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

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

} // namespace android
} // namespace tidy
} // namespace clang
