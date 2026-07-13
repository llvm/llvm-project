//===--- LexerUtilsTest.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/LexerUtils.h"

#include "clang/AST/DeclCXX.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gtest/gtest.h"

namespace clang::tidy::test {

using clang::tooling::FileContentMappings;

static std::unique_ptr<ASTUnit>
buildAST(StringRef Code, const FileContentMappings &Mappings = {}) {
  std::vector<std::string> Args = {"-std=c++20"};
  return clang::tooling::buildASTFromCodeWithArgs(
      Code, Args, "input.cc", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      clang::tooling::getClangStripDependencyFileAdjuster(), Mappings);
}

static CharSourceRange rangeFromAnnotations(const llvm::Annotations &A,
                                            const SourceManager &SM, FileID FID,
                                            llvm::StringRef Name = "") {
  const auto R = A.range(Name);
  const SourceLocation Begin =
      SM.getLocForStartOfFile(FID).getLocWithOffset(R.Begin);
  const SourceLocation End =
      SM.getLocForStartOfFile(FID).getLocWithOffset(R.End);
  return CharSourceRange::getCharRange(Begin, End);
}

static bool isRawIdentifierNamed(const Token &Tok, StringRef Name) {
  return Tok.is(tok::raw_identifier) && Tok.getRawIdentifier() == Name;
}

namespace {

TEST(LexerUtilsTest, GetCommentsInRangeAdjacentComments) {
  llvm::Annotations Code(R"cpp(
void f() {
  $range[[/*first*/ /*second*/]]
  int x = 0;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getCommentsInRange(Range, SM, LangOpts);
  ASSERT_EQ(2u, Comments.size());
  EXPECT_EQ("/*first*/", Comments[0].Text);
  EXPECT_EQ("/*second*/", Comments[1].Text);
  const StringRef CodeText = Code.code();
  const size_t FirstOffset = CodeText.find("/*first*/");
  ASSERT_NE(StringRef::npos, FirstOffset);
  const size_t SecondOffset = CodeText.find("/*second*/");
  ASSERT_NE(StringRef::npos, SecondOffset);
  EXPECT_EQ(FirstOffset, SM.getFileOffset(Comments[0].Loc));
  EXPECT_EQ(SecondOffset, SM.getFileOffset(Comments[1].Loc));
}

TEST(LexerUtilsTest, GetCommentsInRangeKeepsCommentsAcrossTokens) {
  llvm::Annotations Code(R"cpp(
void f() {
  int x = ($range[[/*first*/ 0, /*second*/]] 1);
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getCommentsInRange(Range, SM, LangOpts);
  ASSERT_EQ(2u, Comments.size());
  EXPECT_EQ("/*first*/", Comments[0].Text);
  EXPECT_EQ("/*second*/", Comments[1].Text);
  const StringRef CodeText = Code.code();
  const size_t FirstOffset = CodeText.find("/*first*/");
  ASSERT_NE(StringRef::npos, FirstOffset);
  const size_t SecondOffset = CodeText.find("/*second*/");
  ASSERT_NE(StringRef::npos, SecondOffset);
  EXPECT_EQ(FirstOffset, SM.getFileOffset(Comments[0].Loc));
  EXPECT_EQ(SecondOffset, SM.getFileOffset(Comments[1].Loc));
}

TEST(LexerUtilsTest, GetCommentsInRangeLineComments) {
  llvm::Annotations Code(R"cpp(
void f() {
  $range[[// first
  // second
  ]]
  int x = 0;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getCommentsInRange(Range, SM, LangOpts);
  ASSERT_EQ(2u, Comments.size());
  EXPECT_EQ("// first", Comments[0].Text);
  EXPECT_EQ("// second", Comments[1].Text);
  const StringRef CodeText = Code.code();
  const size_t FirstOffset = CodeText.find("// first");
  ASSERT_NE(StringRef::npos, FirstOffset);
  const size_t SecondOffset = CodeText.find("// second");
  ASSERT_NE(StringRef::npos, SecondOffset);
  EXPECT_EQ(FirstOffset, SM.getFileOffset(Comments[0].Loc));
  EXPECT_EQ(SecondOffset, SM.getFileOffset(Comments[1].Loc));
}

TEST(LexerUtilsTest, GetCommentsInRangeNoComments) {
  llvm::Annotations Code(R"cpp(
void f() {
  int x = $range[[0 + 1]];
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getCommentsInRange(Range, SM, LangOpts);
  EXPECT_TRUE(Comments.empty());
}

TEST(LexerUtilsTest, GetCommentsInRangeInvalidRange) {
  std::unique_ptr<ASTUnit> AST = buildAST("int value = 0;");
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getCommentsInRange(CharSourceRange(), SM, LangOpts);
  EXPECT_TRUE(Comments.empty());
}

TEST(LexerUtilsTest, FindTokenTextInRangeFindsMatch) {
  llvm::Annotations Code(R"cpp(
struct S {
  $range[[explicit   ]] S(int);
};
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange SearchRange =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      SearchRange, SM, LangOpts,
      [](const Token &Tok) { return isRawIdentifierNamed(Tok, "explicit"); });
  ASSERT_TRUE(MatchedRange.isValid());

  const StringRef CodeText = Code.code();
  const size_t ExplicitOffset = CodeText.find("explicit");
  ASSERT_NE(StringRef::npos, ExplicitOffset);
  const size_t ConstructorOffset = CodeText.find("S(int)");
  ASSERT_NE(StringRef::npos, ConstructorOffset);
  EXPECT_EQ(ExplicitOffset, SM.getFileOffset(MatchedRange.getBegin()));
  EXPECT_EQ(ConstructorOffset, SM.getFileOffset(MatchedRange.getEnd()));
}

TEST(LexerUtilsTest, FindTokenTextInRangeReturnsInvalidWhenNotFound) {
  llvm::Annotations Code(R"cpp(
struct S {
  $range[[int x = 0;]]
  S(int);
};
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange SearchRange =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      SearchRange, SM, LangOpts,
      [](const Token &Tok) { return isRawIdentifierNamed(Tok, "explicit"); });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest, FindTokenTextInRangeDoesNotMatchTokenAtEndBoundary) {
  llvm::Annotations Code(R"cpp(
struct S {
  $range[[int x = 0; ]]explicit S(int);
};
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange SearchRange =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      SearchRange, SM, LangOpts,
      [](const Token &Tok) { return isRawIdentifierNamed(Tok, "explicit"); });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest,
     FindTokenTextInRangeReturnsInvalidWhenPredicateNeverMatches) {
  llvm::Annotations Code(R"cpp(
struct S {
  $range[[explicit ]] S(int);
};
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange SearchRange =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      SearchRange, SM, LangOpts, [](const Token &) { return false; });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest, FindTokenTextInRangeReturnsInvalidForInvalidRange) {
  std::unique_ptr<ASTUnit> AST = buildAST("struct S { explicit S(int); };");
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      CharSourceRange(), SM, LangOpts, [](const Token &) { return true; });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest, FindTokenTextInRangeReturnsInvalidForReversedOffsets) {
  llvm::Annotations Code(R"cpp(
struct S {
  $a^explicit S(int);$b^
};
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const SourceLocation MainFileStart =
      SM.getLocForStartOfFile(SM.getMainFileID());
  const SourceLocation Begin = MainFileStart.getLocWithOffset(Code.point("b"));
  const SourceLocation End = MainFileStart.getLocWithOffset(Code.point("a"));
  ASSERT_TRUE(SM.isBeforeInTranslationUnit(End, Begin));

  const CharSourceRange ReversedRange =
      CharSourceRange::getCharRange(Begin, End);
  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      ReversedRange, SM, LangOpts,
      [](const Token &Tok) { return isRawIdentifierNamed(Tok, "explicit"); });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest, FindTokenTextInRangeReturnsInvalidWhenFileRangeIsInvalid) {
  llvm::Annotations Code(R"cpp(
#include "header.h"
int $begin^main_var = 0;
)cpp");
  const FileContentMappings Mappings = {
      {"header.h", "int header_var = 0;\n"},
  };
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code(), Mappings);
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const SourceLocation MainFileStart =
      SM.getLocForStartOfFile(SM.getMainFileID());
  const SourceLocation Begin =
      MainFileStart.getLocWithOffset(Code.point("begin"));
  ASSERT_TRUE(Begin.isFileID());

  auto HeaderFile = AST->getFileManager().getOptionalFileRef("header.h");
  ASSERT_TRUE(HeaderFile.has_value());
  const FileID HeaderFID = SM.translateFile(*HeaderFile);
  ASSERT_TRUE(HeaderFID.isValid());
  const SourceLocation HeaderBegin = SM.getLocForStartOfFile(HeaderFID);
  ASSERT_TRUE(HeaderBegin.isFileID());

  const CharSourceRange SearchRange =
      CharSourceRange::getCharRange(Begin, HeaderBegin);
  const CharSourceRange FileRange =
      Lexer::makeFileCharRange(SearchRange, SM, LangOpts);
  EXPECT_TRUE(FileRange.isInvalid());

  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      SearchRange, SM, LangOpts, [](const Token &) { return true; });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest, FindTokenTextInRangeReturnsInvalidForMacroRange) {
  std::unique_ptr<ASTUnit> AST = buildAST(R"cpp(
#define EXPLICIT explicit
struct S {
  EXPLICIT S(int);
};
)cpp");
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CXXConstructorDecl *Ctor = [&Context] {
    for (const Decl *D : Context.getTranslationUnitDecl()->decls()) {
      const auto *RD = dyn_cast<CXXRecordDecl>(D);
      if (!RD)
        continue;
      for (const CXXConstructorDecl *Ctor : RD->ctors())
        if (!Ctor->isImplicit())
          return Ctor;
    }
    return static_cast<const CXXConstructorDecl *>(nullptr);
  }();
  ASSERT_NE(nullptr, Ctor);
  ASSERT_TRUE(Ctor->getOuterLocStart().isMacroID());

  const CharSourceRange SearchRange = CharSourceRange::getTokenRange(
      Ctor->getOuterLocStart(), Ctor->getEndLoc());
  const CharSourceRange MatchedRange = utils::lexer::findTokenTextInRange(
      SearchRange, SM, LangOpts,
      [](const Token &Tok) { return isRawIdentifierNamed(Tok, "explicit"); });
  EXPECT_TRUE(MatchedRange.isInvalid());
}

TEST(LexerUtilsTest, GetTrailingCommentsInRangeAdjacentComments) {
  llvm::Annotations Code(R"cpp(
void f() {
  $range[[/*first*/ /*second*/]]
  int x = 0;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getTrailingCommentsInRange(Range, SM, LangOpts);
  ASSERT_EQ(2u, Comments.size());
  EXPECT_EQ("/*first*/", Comments[0].Text);
  EXPECT_EQ("/*second*/", Comments[1].Text);
  const StringRef CodeText = Code.code();
  const size_t FirstOffset = CodeText.find("/*first*/");
  ASSERT_NE(StringRef::npos, FirstOffset);
  const size_t SecondOffset = CodeText.find("/*second*/");
  ASSERT_NE(StringRef::npos, SecondOffset);
  EXPECT_EQ(FirstOffset, SM.getFileOffset(Comments[0].Loc));
  EXPECT_EQ(SecondOffset, SM.getFileOffset(Comments[1].Loc));
}

TEST(LexerUtilsTest, GetTrailingCommentsInRangeClearsOnToken) {
  llvm::Annotations Code(R"cpp(
void f() {
  int x = ($range[[/*first*/ 0, /*second*/]] 1);
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getTrailingCommentsInRange(Range, SM, LangOpts);
  ASSERT_EQ(1u, Comments.size());
  EXPECT_EQ("/*second*/", Comments.front().Text);
  const StringRef CodeText = Code.code();
  const size_t SecondOffset = CodeText.find("/*second*/");
  ASSERT_NE(StringRef::npos, SecondOffset);
  EXPECT_EQ(SecondOffset, SM.getFileOffset(Comments.front().Loc));
}

TEST(LexerUtilsTest, GetTrailingCommentsInRangeLineComments) {
  llvm::Annotations Code(R"cpp(
void f() {
  $range[[// first
  // second
  ]]
  int x = 0;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "range");
  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getTrailingCommentsInRange(Range, SM, LangOpts);
  ASSERT_EQ(2u, Comments.size());
  EXPECT_EQ("// first", Comments[0].Text);
  EXPECT_EQ("// second", Comments[1].Text);
  const StringRef CodeText = Code.code();
  const size_t FirstOffset = CodeText.find("// first");
  ASSERT_NE(StringRef::npos, FirstOffset);
  const size_t SecondOffset = CodeText.find("// second");
  ASSERT_NE(StringRef::npos, SecondOffset);
  EXPECT_EQ(FirstOffset, SM.getFileOffset(Comments[0].Loc));
  EXPECT_EQ(SecondOffset, SM.getFileOffset(Comments[1].Loc));
}

TEST(LexerUtilsTest, GetTrailingCommentsInRangeInvalidRange) {
  std::unique_ptr<ASTUnit> AST = buildAST("int value = 0;");
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getTrailingCommentsInRange(CharSourceRange(), SM, LangOpts);
  EXPECT_TRUE(Comments.empty());
}

} // namespace

} // namespace clang::tidy::test
