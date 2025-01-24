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
#include <iostream>
using namespace clang::ast_matchers;

namespace clang::tidy::modernize {
class UseCppStyleCommentsCheck::CStyleCommentHandler : public CommentHandler {
public:
  CStyleCommentHandler(UseCppStyleCommentsCheck &Check)
      : Check(Check),
        CStyleCommentMatch(
            "^[ \t]*/\\*+[ \t\r\n]*(.*[ \t\r\n]*)*[ \t\r\n]*\\*+/[ \t\r\n]*$") {
  }

  std::string convertToCppStyleComment(const SourceManager &SM,
                                       const SourceRange &Range) {
    StringRef CommentText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Range), SM, LangOptions());

    std::string InnerText = CommentText.str();
    InnerText.erase(0, 2);
    InnerText.erase(InnerText.size() - 2, 2);

    std::string Result;
    std::istringstream Stream(InnerText);
    std::string Line;

    if (std::getline(Stream, Line)) {
      size_t startPos = Line.find_first_not_of(" \t");
      if (startPos != std::string::npos) {
        Line = Line.substr(startPos);
      } else {
        Line.clear();
      }
      Result += "// " + Line;
    }

    while (std::getline(Stream, Line)) {
      size_t startPos = Line.find_first_not_of(" \t");
      if (startPos != std::string::npos) {
        Line = Line.substr(startPos);
      } else {
        Line.clear();
      }
      Result += "\n// " + Line;
    }
    return Result;
  }

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    const SourceManager &SM = PP.getSourceManager();

    if (Range.getBegin().isMacroID() || SM.isInSystemHeader(Range.getBegin())) {
      return false;
    }

    const StringRef Text = Lexer::getSourceText(
        CharSourceRange::getCharRange(Range), SM, PP.getLangOpts());

    SmallVector<StringRef> Matches;
    if (!CStyleCommentMatch.match(Text, &Matches)) {
      return false;
    }

    SourceLocation CommentStart = Range.getBegin();
    SourceLocation CommentEnd = Range.getEnd();

    unsigned StartLine = SM.getSpellingLineNumber(CommentStart);
    unsigned EndLine = SM.getSpellingLineNumber(CommentEnd);

    if (StartLine == EndLine) {
      SourceLocation LineBegin =
          SM.translateLineCol(SM.getFileID(CommentStart), StartLine, 1);
      SourceLocation LineEnd =
          SM.translateLineCol(SM.getFileID(CommentEnd), EndLine,
                              std::numeric_limits<unsigned>::max());
      StringRef LineContent = Lexer::getSourceText(
          CharSourceRange::getCharRange(LineBegin, LineEnd), SM,
          PP.getLangOpts());
      size_t CommentStartOffset = SM.getSpellingColumnNumber(CommentStart) - 1;
      StringRef AfterComment =
          LineContent.drop_front(CommentStartOffset + Text.size());

      if (!AfterComment.trim().empty()) {
        return false;
      }
    }

    Check.diag(
        Range.getBegin(),
        "use C++ style comments '//' instead of C style comments '/*...*/'")
        << clang::FixItHint::CreateReplacement(
               Range, convertToCppStyleComment(SM, Range));

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
