//===--- TodoCommentCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TodoCommentCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include <array>
#include <optional>

namespace clang::tidy::google::readability {

class TodoCommentCheck::TodoCommentHandler : public CommentHandler {
public:
  TodoCommentHandler(TodoCommentCheck &Check, std::optional<std::string> User)
      : Check(Check), User(User ? *User : "unknown"),
        TodoMatches{llvm::Regex("^// TODO: (.*) - (.*)$"),
                    llvm::Regex("^// *TODO *(\\(.*\\))?:? ?(.*)$")} {}

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    StringRef Text =
        Lexer::getSourceText(CharSourceRange::getCharRange(Range),
                             PP.getSourceManager(), PP.getLangOpts());

    bool Found = false;
    SmallVector<StringRef, 4> Matches;
    for (const llvm::Regex &TodoMatch : TodoMatches) {
      if (TodoMatch.match(Text, &Matches)) {
        Found = true;
        break;
      }
    }
    if (!Found)
      return false;

    StringRef Info = Matches[1];
    StringRef Comment = Matches[2];

    if (!Info.empty())
      return false;

    std::string NewText = ("// TODO(" + Twine(User) + "): " + Comment).str();

    Check.diag(Range.getBegin(), "missing username/bug in TODO")
        << FixItHint::CreateReplacement(CharSourceRange::getCharRange(Range),
                                        NewText);
    return false;
  }

private:
  TodoCommentCheck &Check;
  std::string User;
  std::array<llvm::Regex, 2> TodoMatches;
};

TodoCommentCheck::TodoCommentCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Handler(std::make_unique<TodoCommentHandler>(
          *this, Context->getOptions().User)) {}

TodoCommentCheck::~TodoCommentCheck() = default;

void TodoCommentCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  PP->addCommentHandler(Handler.get());
}

} // namespace clang::tidy::google::readability
