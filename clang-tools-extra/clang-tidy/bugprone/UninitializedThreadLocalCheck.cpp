//===--- UninitializedThreadLocalCheck.cpp - clang-tidy
//--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UninitializedThreadLocalCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

UninitializedThreadLocalCheck::UninitializedThreadLocalCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void UninitializedThreadLocalCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      declRefExpr(
          // Fast check -- bail out quickly before slower filters
          to(varDecl(hasThreadStorageDuration(),
                     hasDeclContext(functionDecl()))),
          forCallable(decl().bind("ctx")),
          to(varDecl(unless(hasDeclContext(equalsBoundNode("ctx"))))))
          .bind("declref"),
      this);
}

void UninitializedThreadLocalCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<DeclRefExpr>("declref");
  diag(E->getLocation(),
       "variable '%0' might not have been initialized on the current thread. "
       "To guarantee prior initialization on the same thread that performs the "
       "access, consider capturing the address of the variable in the same "
       "block as its initialization, then use the captured address to the "
       "desired code.")
      << E;
}

} // namespace clang::tidy::bugprone
