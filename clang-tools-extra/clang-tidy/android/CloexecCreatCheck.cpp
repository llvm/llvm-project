//===--- CloexecCreatCheck.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
