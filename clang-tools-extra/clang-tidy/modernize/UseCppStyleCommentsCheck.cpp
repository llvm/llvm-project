//===--- UseCppStyleCommentsCheck.cpp - clang-tidy-------------------------===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseCppStyleCommentsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {
class UseCppStyleCommentsCheck::CStyleCommentHandler : public CommentHandler {
public:
  CStyleCommentHandler(UseCppStyleCommentsCheck &Check)
      : Check(Check),
        CStyleCommentMatch(
            "^[ \t]*/\\*+[ \t\n]*(.*[ \t\n]*)*[ \t\n]*\\*+/[ \t\n]*$") {}

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    if (Range.getBegin().isMacroID() ||
        PP.getSourceManager().isInSystemHeader(Range.getBegin()))
      return false;

    const StringRef Text =
        Lexer::getSourceText(CharSourceRange::getCharRange(Range),
                             PP.getSourceManager(), PP.getLangOpts());

    SmallVector<StringRef> Matches;
    if (!CStyleCommentMatch.match(Text, &Matches)) {
      return false;
    }

    Check.diag(
        Range.getBegin(),
        "use C++ style comments '//' instead of C style comments '/*...*/'");

    return false;
  }

private:
  UseCppStyleCommentsCheck &Check;
  llvm::Regex CStyleCommentMatch;
};

UseCppStyleCommentsCheck::UseCppStyleCommentsCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Handler(std::make_unique<CStyleCommentHandler>(*this)) {}

void UseCppStyleCommentsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addCommentHandler(Handler.get());
}

void UseCppStyleCommentsCheck::check(const MatchFinder::MatchResult &Result) {}

UseCppStyleCommentsCheck::~UseCppStyleCommentsCheck() = default;
} // namespace clang::tidy::modernize
