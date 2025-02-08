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
#include <sstream>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {
class UseCppStyleCommentsCheck::CStyleCommentHandler : public CommentHandler {
public:
  CStyleCommentHandler(UseCppStyleCommentsCheck &Check, bool ExcludeDoxygen)
      : Check(Check), ExcludeDoxygen(ExcludeDoxygen),
        CStyleCommentMatch(
            "^[ \t]*/\\*+[ \t\r\n]*(.*[ \t\r\n]*)*[ \t\r\n]*\\*+/[ \t\r\n]*$") {
  }

  void setExcludeDoxygen(bool Exclude) { ExcludeDoxygen = Exclude; }

  bool isExcludeDoxygen() const { return ExcludeDoxygen; }

  std::string convertToCppStyleComment(const SourceManager &SM,
                                       const SourceRange &Range) {
    const StringRef CommentText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Range), SM, LangOptions());

    std::string InnerText = CommentText.str();
    InnerText.erase(0, 2);
    InnerText.erase(InnerText.size() - 2, 2);

    std::string Result;
    std::istringstream Stream(InnerText);
    std::string Line;

    if (std::getline(Stream, Line)) {
      const size_t StartPos = Line.find_first_not_of(" \t");
      if (StartPos != std::string::npos) {
        Line = Line.substr(StartPos);
      } else {
        Line.clear();
      }
      Result += "// " + Line;
    }

    while (std::getline(Stream, Line)) {
      const size_t StartPos = Line.find_first_not_of(" \t");
      if (StartPos != std::string::npos) {
        Line = Line.substr(StartPos);
      } else {
        Line.clear();
      }
      Result += "\n// " + Line;
    }
    return Result;
  }

  bool isDoxygenStyleComment(const StringRef &Text) {
    StringRef Trimmed = Text.ltrim();
    return Trimmed.starts_with("/**") || Trimmed.starts_with("/*!") ||
           Trimmed.starts_with("///") || Trimmed.starts_with("//!") ||
           (Trimmed.starts_with("/*") &&
            Trimmed.drop_front(2).starts_with("*"));
  }

  bool CheckForTextAfterComment(Preprocessor &PP, SourceRange Range) {
    const SourceManager &SM = PP.getSourceManager();
    const SourceLocation CommentEnd = Range.getEnd();

    unsigned EndLine = SM.getSpellingLineNumber(CommentEnd);
    unsigned EndCol = SM.getSpellingColumnNumber(CommentEnd);

    const SourceLocation LineBegin =
        SM.translateLineCol(SM.getFileID(CommentEnd), EndLine, EndCol);
    const SourceLocation LineEnd =
        SM.translateLineCol(SM.getFileID(CommentEnd), EndLine,
                            std::numeric_limits<unsigned>::max());
    const StringRef AfterComment =
        Lexer::getSourceText(CharSourceRange::getCharRange(LineBegin, LineEnd),
                             SM, PP.getLangOpts());

    return !AfterComment.trim().empty();
  }

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    const SourceManager &SM = PP.getSourceManager();

    if (Range.getBegin().isMacroID() || SM.isInSystemHeader(Range.getBegin())) {
      return false;
    }

    const StringRef Text = Lexer::getSourceText(
        CharSourceRange::getCharRange(Range), SM, PP.getLangOpts());

    if (ExcludeDoxygen && isDoxygenStyleComment(Text)) {
      return false;
    }

    SmallVector<StringRef> Matches;
    if (!CStyleCommentMatch.match(Text, &Matches)) {
      return false;
    }

    if (CheckForTextAfterComment(PP, Range)) {
      return false;
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
  bool ExcludeDoxygen;
  llvm::Regex CStyleCommentMatch;
};

UseCppStyleCommentsCheck::UseCppStyleCommentsCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Handler(std::make_unique<CStyleCommentHandler>(
          *this, Options.get("ExcludeDoxygenStyleComments", false))) {}

void UseCppStyleCommentsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addCommentHandler(Handler.get());
}

void UseCppStyleCommentsCheck::check(const MatchFinder::MatchResult &Result) {}

void UseCppStyleCommentsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ExcludeDoxygenStyleComments",
                Handler->isExcludeDoxygen());
}

UseCppStyleCommentsCheck::~UseCppStyleCommentsCheck() = default;
} // namespace clang::tidy::readability
