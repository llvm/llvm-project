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
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {
class UseCppStyleCommentsCheck::CStyleCommentHandler : public CommentHandler {
public:
  CStyleCommentHandler(UseCppStyleCommentsCheck &Check, bool ExcludeDoxygen,
                       StringRef ExcludedComments)
      : Check(Check), ExcludeDoxygen(ExcludeDoxygen),
        ExcludedComments(ExcludedComments),
        ExcludedCommentMatch(ExcludedComments),
        CStyleCommentMatch(
            "^[ \t]*/\\*+[ \t\r\n]*(.*[ \t\r\n]*)*[ \t\r\n]*\\*+/[ \t\r\n]*$") {
  }

  StringRef getExcludedCommentsRegex() const { return ExcludedComments; }

  bool isExcludeDoxygen() const { return ExcludeDoxygen; }

  void convertToCppStyleCommentFixes(const SourceManager &SM,
                                     const SourceRange &Range,
                                     SmallVectorImpl<FixItHint> &FixIts) {

    StringRef Raw = Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                                         SM, LangOptions());

    size_t FirstStar = Raw.find('*');
    size_t LastStar = Raw.rfind('*');

    StringRef CommentText;
    if (FirstStar != StringRef::npos && LastStar != StringRef::npos &&
        LastStar > FirstStar) {
      CommentText = Raw.substr(FirstStar + 1, LastStar - FirstStar - 1);
    }

    CommentText = CommentText.trim(" \t\r\n*/");

    SmallVector<StringRef, 8> Lines;
    CommentText.split(Lines, '\n');

    const FileID FID = SM.getFileID(Range.getBegin());
    unsigned LineNo = SM.getSpellingLineNumber(Range.getBegin());

    for (auto &Line : Lines) {
      Line = Line.ltrim(" \t");
      SourceLocation LineStart = SM.translateLineCol(FID, LineNo, 1);
      SourceLocation LineEnd =
          Lexer::getLocForEndOfToken(Range.getEnd(), 0, SM, LangOptions());

      FixIts.push_back(FixItHint::CreateReplacement(
          CharSourceRange::getCharRange(LineStart, LineEnd),
          "// " + Line.str()));

      ++LineNo;
    }
  }

  bool isDoxygenStyleComment(const StringRef &Text) {
    StringRef Trimmed = Text.ltrim();
    return Trimmed.starts_with("/**") || Trimmed.starts_with("/*!") ||
           Trimmed.starts_with("///") || Trimmed.starts_with("//!") ||
           (Trimmed.starts_with("/*") &&
            Trimmed.drop_front(2).starts_with("*"));
  }

  bool CheckForCodeAfterComment(Preprocessor &PP, SourceRange Range) {
    const SourceManager &SM = PP.getSourceManager();
    const SourceLocation CommentStart = Range.getBegin(),
                         CommentEnd = Range.getEnd();
    const std::optional<Token> NextTok =
        Lexer::findNextToken(CommentStart, SM, PP.getLangOpts());
    if (!NextTok.has_value())
      return false;

    const std::string tokenSpelling =
        Lexer::getSpelling(*NextTok, SM, PP.getLangOpts());
    const unsigned lineNo = SM.getSpellingLineNumber(CommentEnd);
    const SourceLocation loc = NextTok->getLocation();
    const unsigned tokenLine = SM.getSpellingLineNumber(loc);
    return lineNo == tokenLine;
  }

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    const SourceManager &SM = PP.getSourceManager();

    if (Range.getBegin().isMacroID() || SM.isInSystemHeader(Range.getBegin()))
      return false;

    const StringRef Text = Lexer::getSourceText(
        CharSourceRange::getCharRange(Range), SM, PP.getLangOpts());

    if (!ExcludedCommentMatch.isValid()) {
      llvm::errs() << "Warning: Invalid regex pattern:" << ExcludedComments
                   << ":for ExcludedComments\n";
    } else if (ExcludedCommentMatch.match(Text)) {
      return false;
    }
    if (ExcludeDoxygen && isDoxygenStyleComment(Text))
      return false;

    SmallVector<StringRef> Matches;
    if (!CStyleCommentMatch.match(Text, &Matches))
      return false;

    if (CheckForCodeAfterComment(PP, Range))
      return false;

    SmallVector<FixItHint, 4> FixIts;
    convertToCppStyleCommentFixes(SM, Range, FixIts);

    auto D = Check.diag(
        Range.getBegin(),
        "use C++ style comments '//' instead of C style comments '/*...*/'");

    for (const auto &Fix : FixIts)
      D << Fix;

    return false;
  }

private:
  UseCppStyleCommentsCheck &Check;
  bool ExcludeDoxygen;
  StringRef ExcludedComments;
  llvm::Regex ExcludedCommentMatch;
  llvm::Regex CStyleCommentMatch;
};

UseCppStyleCommentsCheck::UseCppStyleCommentsCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Handler(std::make_unique<CStyleCommentHandler>(
          *this, Options.get("ExcludeDoxygenStyleComments", false),
          Options.get("ExcludedComments", "^$"))) {}

void UseCppStyleCommentsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ExcludeDoxygenStyleComments",
                Handler->isExcludeDoxygen());
  Options.store(Opts, "ExcludedComments", Handler->getExcludedCommentsRegex());
}
void UseCppStyleCommentsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addCommentHandler(Handler.get());
}

void UseCppStyleCommentsCheck::check(const MatchFinder::MatchResult &Result) {}

UseCppStyleCommentsCheck::~UseCppStyleCommentsCheck() = default;
} // namespace clang::tidy::readability
