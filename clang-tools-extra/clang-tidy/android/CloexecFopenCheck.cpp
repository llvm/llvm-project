//===--- CloexecFopenCheck.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.  //
//===----------------------------------------------------------------------===//

#include "CloexecFopenCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecFopenCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));
  registerMatchersImpl(Finder,
                       functionDecl(isExternC(), returns(asString("FILE *")),
                                    hasName("fopen"),
                                    hasParameter(0, CharPointerType),
                                    hasParameter(1, CharPointerType)));
}

void CloexecFopenCheck::check(const MatchFinder::MatchResult &Result) {
  insertStringFlag(Result, /*Mode=*/'e', /*ArgPos=*/1);
}

} // namespace android
} // namespace tidy
} // namespace clang
