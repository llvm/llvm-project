//===--- LexerUtilsTest.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/LexerUtils.h"

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

namespace {

TEST(LexerUtilsTest, GetSourceText) {
  llvm::Annotations Code(R"cpp(
int main() {
  [[int value = 42;]]
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID());

  EXPECT_EQ("int value = 42;",
            utils::lexer::getSourceText(Range, SM, LangOpts));
}

TEST(LexerUtilsTest, GetSourceTextInvalidRange) {
  std::unique_ptr<ASTUnit> AST = buildAST("int value = 0;");
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  EXPECT_TRUE(
      utils::lexer::getSourceText(CharSourceRange(), SM, LangOpts).empty());
}

TEST(LexerUtilsTest, GetSourceTextCrossFileRange) {
  const char *Code = R"cpp(
#include "header.h"
int main() { return value; }
)cpp";
  FileContentMappings Mappings = {{"header.h", "int value;\n"}};
  std::unique_ptr<ASTUnit> AST = buildAST(Code, Mappings);
  ASSERT_TRUE(AST);

  ASTContext &Context = AST->getASTContext();
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const SourceLocation MainBegin = SM.getLocForStartOfFile(SM.getMainFileID());
  llvm::Expected<FileEntryRef> HeaderRef =
      SM.getFileManager().getFileRef("header.h");
  ASSERT_TRUE(static_cast<bool>(HeaderRef));
  const FileID HeaderID = SM.getOrCreateFileID(*HeaderRef, SrcMgr::C_User);
  const SourceLocation HeaderBegin = SM.getLocForStartOfFile(HeaderID);
  ASSERT_TRUE(HeaderBegin.isValid());

  const CharSourceRange CrossRange =
      CharSourceRange::getCharRange(MainBegin, HeaderBegin);
  EXPECT_TRUE(utils::lexer::getSourceText(CrossRange, SM, LangOpts).empty());
}

TEST(LexerUtilsTest, AnalyzeTokenRangeInvalidRange) {
  std::unique_ptr<ASTUnit> AST = buildAST("int value = 0;");
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const utils::lexer::TokenRangeInfo Info =
      utils::lexer::analyzeTokenRange(CharSourceRange(), SM, LangOpts);
  EXPECT_FALSE(Info.HasComment);
  EXPECT_FALSE(Info.HasIdentifier);
  EXPECT_FALSE(Info.HasPointerOrRef);
}

TEST(LexerUtilsTest, AnalyzeTokenRangeCommentOnly) {
  llvm::Annotations Code(R"cpp(
void f() {
  [[/*comment*/]]
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID());
  const utils::lexer::TokenRangeInfo Info =
      utils::lexer::analyzeTokenRange(Range, SM, LangOpts);
  EXPECT_TRUE(Info.HasComment);
  EXPECT_FALSE(Info.HasIdentifier);
  EXPECT_FALSE(Info.HasPointerOrRef);
}

TEST(LexerUtilsTest, AnalyzeTokenRangePointerAndReference) {
  llvm::Annotations Code(R"cpp(
void f() {
  int $ptr[[*]]Ptr;
  int $ref[[&]]Ref = *Ptr;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange PtrRange =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "ptr");
  const utils::lexer::TokenRangeInfo PtrInfo =
      utils::lexer::analyzeTokenRange(PtrRange, SM, LangOpts);
  EXPECT_TRUE(PtrInfo.HasPointerOrRef);
  EXPECT_FALSE(PtrInfo.HasIdentifier);
  EXPECT_FALSE(PtrInfo.HasComment);

  const CharSourceRange RefRange =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "ref");
  const utils::lexer::TokenRangeInfo RefInfo =
      utils::lexer::analyzeTokenRange(RefRange, SM, LangOpts);
  EXPECT_TRUE(RefInfo.HasPointerOrRef);
  EXPECT_FALSE(RefInfo.HasIdentifier);
  EXPECT_FALSE(RefInfo.HasComment);
}

TEST(LexerUtilsTest, AnalyzeTokenRangeIdentifier) {
  llvm::Annotations Code(R"cpp(
void f() {
  int $id[[Name]] = 0;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "id");
  const utils::lexer::TokenRangeInfo Info =
      utils::lexer::analyzeTokenRange(Range, SM, LangOpts);
  EXPECT_FALSE(Info.HasComment);
  EXPECT_TRUE(Info.HasIdentifier);
  EXPECT_FALSE(Info.HasPointerOrRef);
}

TEST(LexerUtilsTest, AnalyzeTokenRangeIdentifierKeyword) {
  llvm::Annotations Code(R"cpp(
$kw[[struct]] S {};
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "kw");
  const utils::lexer::TokenRangeInfo Info =
      utils::lexer::analyzeTokenRange(Range, SM, LangOpts);
  EXPECT_FALSE(Info.HasComment);
  EXPECT_TRUE(Info.HasIdentifier);
  EXPECT_FALSE(Info.HasPointerOrRef);
}

TEST(LexerUtilsTest, AnalyzeTokenRangeLogicalAnd) {
  llvm::Annotations Code(R"cpp(
void f(bool a, bool b) {
  bool c = a $and[[&&]] b;
}
)cpp");
  std::unique_ptr<ASTUnit> AST = buildAST(Code.code());
  ASSERT_TRUE(AST);
  const ASTContext &Context = AST->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const CharSourceRange Range =
      rangeFromAnnotations(Code, SM, SM.getMainFileID(), "and");
  const utils::lexer::TokenRangeInfo Info =
      utils::lexer::analyzeTokenRange(Range, SM, LangOpts);
  EXPECT_FALSE(Info.HasComment);
  EXPECT_FALSE(Info.HasIdentifier);
  EXPECT_FALSE(Info.HasPointerOrRef);
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
