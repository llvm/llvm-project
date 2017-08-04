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
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

void CloexecCreatCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));
  auto MODETType = hasType(namedDecl(hasName("mode_t")));

  Finder->addMatcher(
      callExpr(callee(functionDecl(isExternC(), returns(isInteger()),
                                   hasName("creat"),
                                   hasParameter(0, CharPointerType),
                                   hasParameter(1, MODETType))
                          .bind("funcDecl")))
          .bind("creatFn"),
      this);
}

void CloexecCreatCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("creatFn");
  const SourceManager &SM = *Result.SourceManager;

  const std::string &ReplacementText =
      (Twine("open (") +
       Lexer::getSourceText(CharSourceRange::getTokenRange(
                                MatchedCall->getArg(0)->getSourceRange()),
                            SM, Result.Context->getLangOpts()) +
       ", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, " +
       Lexer::getSourceText(CharSourceRange::getTokenRange(
                                MatchedCall->getArg(1)->getSourceRange()),
                            SM, Result.Context->getLangOpts()) +
       ")")
          .str();

  diag(MatchedCall->getLocStart(),
       "prefer open() to creat() because open() allows O_CLOEXEC")
      << FixItHint::CreateReplacement(MatchedCall->getSourceRange(),
                                      ReplacementText);
}

} // namespace android
} // namespace tidy
} // namespace clang
