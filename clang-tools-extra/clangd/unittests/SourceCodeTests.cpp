//===-- SourceCodeTests.cpp  ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "support/Context.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <tuple>

namespace clang {
namespace clangd {
namespace {

using llvm::Failed;
using llvm::FailedWithMessage;
using llvm::HasValue;

MATCHER_P2(Pos, Line, Col, "") {
  return arg.line == int(Line) && arg.character == int(Col);
}

MATCHER_P(macroName, Name, "") { return arg.Name == Name; }

/// A helper to make tests easier to read.
Position position(int Line, int Character) {
  Position Pos;
  Pos.line = Line;
  Pos.character = Character;
  return Pos;
}

TEST(SourceCodeTests, lspLength) {
  EXPECT_EQ(lspLength(""), 0UL);
  EXPECT_EQ(lspLength("ascii"), 5UL);
  // BMP
  EXPECT_EQ(lspLength("↓"), 1UL);
  EXPECT_EQ(lspLength("¥"), 1UL);
  // astral
  EXPECT_EQ(lspLength("😂"), 2UL);

  WithContextValue UTF8(kCurrentOffsetEncoding, OffsetEncoding::UTF8);
  EXPECT_EQ(lspLength(""), 0UL);
  EXPECT_EQ(lspLength("ascii"), 5UL);
  // BMP
  EXPECT_EQ(lspLength("↓"), 3UL);
  EXPECT_EQ(lspLength("¥"), 2UL);
  // astral
  EXPECT_EQ(lspLength("😂"), 4UL);

  WithContextValue UTF32(kCurrentOffsetEncoding, OffsetEncoding::UTF32);
  EXPECT_EQ(lspLength(""), 0UL);
  EXPECT_EQ(lspLength("ascii"), 5UL);
  // BMP
  EXPECT_EQ(lspLength("↓"), 1UL);
  EXPECT_EQ(lspLength("¥"), 1UL);
  // astral
  EXPECT_EQ(lspLength("😂"), 1UL);
}

TEST(SourceCodeTests, lspLengthBadUTF8) {
  // Results are not well-defined if source file isn't valid UTF-8.
  // However we shouldn't crash or return something totally wild.
  const char *BadUTF8[] = {"\xa0", "\xff\xff\xff\xff\xff"};

  for (OffsetEncoding Encoding :
       {OffsetEncoding::UTF8, OffsetEncoding::UTF16, OffsetEncoding::UTF32}) {
    WithContextValue UTF32(kCurrentOffsetEncoding, Encoding);
    for (const char *Bad : BadUTF8) {
      EXPECT_GE(lspLength(Bad), 0u);
      EXPECT_LE(lspLength(Bad), strlen(Bad));
    }
  }
}

// The = → 🡆 below are ASCII (1 byte), BMP (3 bytes), and astral (4 bytes).
const char File[] = R"(0:0 = 0
1:0 → 8
2:0 🡆 18)";
struct Line {
  unsigned Number;
  unsigned Offset;
  unsigned Length;
};
Line FileLines[] = {Line{0, 0, 7}, Line{1, 8, 9}, Line{2, 18, 11}};

TEST(SourceCodeTests, PositionToOffset) {
  // line out of bounds
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(-1, 2)), llvm::Failed());
  // first line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, -1)),
                       llvm::Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 0)),
                       llvm::HasValue(0)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 3)),
                       llvm::HasValue(3)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 6)),
                       llvm::HasValue(6)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 7)),
                       llvm::HasValue(7)); // the newline itself
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 7), false),
                       llvm::HasValue(7));
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 8)),
                       llvm::HasValue(7)); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 8), false),
                       llvm::Failed()); // out of range
  // middle line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, -1)),
                       llvm::Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 0)),
                       llvm::HasValue(8)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 3)),
                       llvm::HasValue(11)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 3), false),
                       llvm::HasValue(11));
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 6)),
                       llvm::HasValue(16)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 7)),
                       llvm::HasValue(17)); // the newline itself
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 8)),
                       llvm::HasValue(17)); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 8), false),
                       llvm::Failed()); // out of range
  // last line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, -1)),
                       llvm::Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 0)),
                       llvm::HasValue(18)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 3)),
                       llvm::HasValue(21)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 5), false),
                       llvm::Failed()); // middle of surrogate pair
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 5)),
                       llvm::HasValue(26)); // middle of surrogate pair
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 6), false),
                       llvm::HasValue(26)); // end of surrogate pair
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 8)),
                       llvm::HasValue(28)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 9)),
                       llvm::HasValue(29)); // EOF
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 10), false),
                       llvm::Failed()); // out of range
  // line out of bounds
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 0)), llvm::Failed());
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 1)), llvm::Failed());

  // Codepoints are similar, except near astral characters.
  WithContextValue UTF32(kCurrentOffsetEncoding, OffsetEncoding::UTF32);
  // line out of bounds
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(-1, 2)), llvm::Failed());
  // first line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, -1)),
                       llvm::Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 0)),
                       llvm::HasValue(0)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 3)),
                       llvm::HasValue(3)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 6)),
                       llvm::HasValue(6)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 7)),
                       llvm::HasValue(7)); // the newline itself
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 7), false),
                       llvm::HasValue(7));
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 8)),
                       llvm::HasValue(7)); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 8), false),
                       llvm::Failed()); // out of range
  // middle line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, -1)),
                       llvm::Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 0)),
                       llvm::HasValue(8)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 3)),
                       llvm::HasValue(11)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 3), false),
                       llvm::HasValue(11));
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 6)),
                       llvm::HasValue(16)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 7)),
                       llvm::HasValue(17)); // the newline itself
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 8)),
                       llvm::HasValue(17)); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 8), false),
                       llvm::Failed()); // out of range
  // last line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, -1)),
                       llvm::Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 0)),
                       llvm::HasValue(18)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 4)),
                       llvm::HasValue(22)); // Before astral character.
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 5), false),
                       llvm::HasValue(26)); // after astral character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 7)),
                       llvm::HasValue(28)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 8)),
                       llvm::HasValue(29)); // EOF
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 9), false),
                       llvm::Failed()); // out of range
  // line out of bounds
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 0)), llvm::Failed());
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 1)), llvm::Failed());

  // Test UTF-8, where transformations are trivial.
  WithContextValue UTF8(kCurrentOffsetEncoding, OffsetEncoding::UTF8);
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(-1, 2)), llvm::Failed());
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 0)), llvm::Failed());
  for (Line L : FileLines) {
    EXPECT_THAT_EXPECTED(positionToOffset(File, position(L.Number, -1)),
                         llvm::Failed()); // out of range
    for (unsigned I = 0; I <= L.Length; ++I)
      EXPECT_THAT_EXPECTED(positionToOffset(File, position(L.Number, I)),
                           llvm::HasValue(L.Offset + I));
    EXPECT_THAT_EXPECTED(
        positionToOffset(File, position(L.Number, L.Length + 1)),
        llvm::HasValue(L.Offset + L.Length));
    EXPECT_THAT_EXPECTED(
        positionToOffset(File, position(L.Number, L.Length + 1), false),
        llvm::Failed()); // out of range
  }
}

TEST(SourceCodeTests, OffsetToPosition) {
  EXPECT_THAT(offsetToPosition(File, 0), Pos(0, 0)) << "start of file";
  EXPECT_THAT(offsetToPosition(File, 3), Pos(0, 3)) << "in first line";
  EXPECT_THAT(offsetToPosition(File, 6), Pos(0, 6)) << "end of first line";
  EXPECT_THAT(offsetToPosition(File, 7), Pos(0, 7)) << "first newline";
  EXPECT_THAT(offsetToPosition(File, 8), Pos(1, 0)) << "start of second line";
  EXPECT_THAT(offsetToPosition(File, 12), Pos(1, 4)) << "before BMP char";
  EXPECT_THAT(offsetToPosition(File, 13), Pos(1, 5)) << "in BMP char";
  EXPECT_THAT(offsetToPosition(File, 15), Pos(1, 5)) << "after BMP char";
  EXPECT_THAT(offsetToPosition(File, 16), Pos(1, 6)) << "end of second line";
  EXPECT_THAT(offsetToPosition(File, 17), Pos(1, 7)) << "second newline";
  EXPECT_THAT(offsetToPosition(File, 18), Pos(2, 0)) << "start of last line";
  EXPECT_THAT(offsetToPosition(File, 21), Pos(2, 3)) << "in last line";
  EXPECT_THAT(offsetToPosition(File, 22), Pos(2, 4)) << "before astral char";
  EXPECT_THAT(offsetToPosition(File, 24), Pos(2, 6)) << "in astral char";
  EXPECT_THAT(offsetToPosition(File, 26), Pos(2, 6)) << "after astral char";
  EXPECT_THAT(offsetToPosition(File, 28), Pos(2, 8)) << "end of last line";
  EXPECT_THAT(offsetToPosition(File, 29), Pos(2, 9)) << "EOF";
  EXPECT_THAT(offsetToPosition(File, 30), Pos(2, 9)) << "out of bounds";

  // Codepoints are similar, except near astral characters.
  WithContextValue UTF32(kCurrentOffsetEncoding, OffsetEncoding::UTF32);
  EXPECT_THAT(offsetToPosition(File, 0), Pos(0, 0)) << "start of file";
  EXPECT_THAT(offsetToPosition(File, 3), Pos(0, 3)) << "in first line";
  EXPECT_THAT(offsetToPosition(File, 6), Pos(0, 6)) << "end of first line";
  EXPECT_THAT(offsetToPosition(File, 7), Pos(0, 7)) << "first newline";
  EXPECT_THAT(offsetToPosition(File, 8), Pos(1, 0)) << "start of second line";
  EXPECT_THAT(offsetToPosition(File, 12), Pos(1, 4)) << "before BMP char";
  EXPECT_THAT(offsetToPosition(File, 13), Pos(1, 5)) << "in BMP char";
  EXPECT_THAT(offsetToPosition(File, 15), Pos(1, 5)) << "after BMP char";
  EXPECT_THAT(offsetToPosition(File, 16), Pos(1, 6)) << "end of second line";
  EXPECT_THAT(offsetToPosition(File, 17), Pos(1, 7)) << "second newline";
  EXPECT_THAT(offsetToPosition(File, 18), Pos(2, 0)) << "start of last line";
  EXPECT_THAT(offsetToPosition(File, 21), Pos(2, 3)) << "in last line";
  EXPECT_THAT(offsetToPosition(File, 22), Pos(2, 4)) << "before astral char";
  EXPECT_THAT(offsetToPosition(File, 24), Pos(2, 5)) << "in astral char";
  EXPECT_THAT(offsetToPosition(File, 26), Pos(2, 5)) << "after astral char";
  EXPECT_THAT(offsetToPosition(File, 28), Pos(2, 7)) << "end of last line";
  EXPECT_THAT(offsetToPosition(File, 29), Pos(2, 8)) << "EOF";
  EXPECT_THAT(offsetToPosition(File, 30), Pos(2, 8)) << "out of bounds";

  WithContextValue UTF8(kCurrentOffsetEncoding, OffsetEncoding::UTF8);
  for (Line L : FileLines) {
    for (unsigned I = 0; I <= L.Length; ++I)
      EXPECT_THAT(offsetToPosition(File, L.Offset + I), Pos(L.Number, I));
  }
  EXPECT_THAT(offsetToPosition(File, 30), Pos(2, 11)) << "out of bounds";
}

TEST(SourceCodeTests, SourceLocationInMainFile) {
  Annotations Source(R"cpp(
    ^in^t ^foo
    ^bar
    ^baz ^() {}  {} {} {} { }^
)cpp");

  SourceManagerForFile Owner("foo.cpp", Source.code());
  SourceManager &SM = Owner.get();

  SourceLocation StartOfFile = SM.getLocForStartOfFile(SM.getMainFileID());
  EXPECT_THAT_EXPECTED(sourceLocationInMainFile(SM, position(0, 0)),
                       HasValue(StartOfFile));
  // End of file.
  EXPECT_THAT_EXPECTED(
      sourceLocationInMainFile(SM, position(4, 0)),
      HasValue(StartOfFile.getLocWithOffset(Source.code().size())));
  // Column number is too large.
  EXPECT_THAT_EXPECTED(sourceLocationInMainFile(SM, position(0, 1)), Failed());
  EXPECT_THAT_EXPECTED(sourceLocationInMainFile(SM, position(0, 100)),
                       Failed());
  EXPECT_THAT_EXPECTED(sourceLocationInMainFile(SM, position(4, 1)), Failed());
  // Line number is too large.
  EXPECT_THAT_EXPECTED(sourceLocationInMainFile(SM, position(5, 0)), Failed());
  // Check all positions mentioned in the test return valid results.
  for (auto P : Source.points()) {
    size_t Offset = llvm::cantFail(positionToOffset(Source.code(), P));
    EXPECT_THAT_EXPECTED(sourceLocationInMainFile(SM, P),
                         HasValue(StartOfFile.getLocWithOffset(Offset)));
  }
}

TEST(SourceCodeTests, isReservedName) {
  EXPECT_FALSE(isReservedName(""));
  EXPECT_FALSE(isReservedName("_"));
  EXPECT_FALSE(isReservedName("foo"));
  EXPECT_FALSE(isReservedName("_foo"));
  EXPECT_TRUE(isReservedName("__foo"));
  EXPECT_TRUE(isReservedName("_Foo"));
  EXPECT_FALSE(isReservedName("foo__bar")) << "FIXME";
}

TEST(SourceCodeTests, CollectIdentifiers) {
  auto Style = format::getLLVMStyle();
  auto IDs = collectIdentifiers(R"cpp(
  #include "a.h"
  void foo() { int xyz; int abc = xyz; return foo(); }
  )cpp",
                                Style);
  EXPECT_EQ(IDs.size(), 7u);
  EXPECT_EQ(IDs["include"], 1u);
  EXPECT_EQ(IDs["void"], 1u);
  EXPECT_EQ(IDs["int"], 2u);
  EXPECT_EQ(IDs["xyz"], 2u);
  EXPECT_EQ(IDs["abc"], 1u);
  EXPECT_EQ(IDs["return"], 1u);
  EXPECT_EQ(IDs["foo"], 2u);
}

TEST(SourceCodeTests, CollectWords) {
  auto Words = collectWords(R"cpp(
  #define FIZZ_BUZZ
  // this is a comment
  std::string getSomeText() { return "magic word"; }
  )cpp");
  std::set<StringRef> ActualWords(Words.keys().begin(), Words.keys().end());
  std::set<StringRef> ExpectedWords = {"define",  "fizz",   "buzz", "this",
                                       "comment", "string", "some", "text",
                                       "return",  "magic",  "word"};
  EXPECT_EQ(ActualWords, ExpectedWords);
}

class SpelledWordsTest : public ::testing::Test {
  llvm::Optional<ParsedAST> AST;

  llvm::Optional<SpelledWord> tryWord(const char *Text) {
    llvm::Annotations A(Text);
    auto TU = TestTU::withCode(A.code());
    AST = TU.build();
    auto SW = SpelledWord::touching(
        AST->getSourceManager().getComposedLoc(
            AST->getSourceManager().getMainFileID(), A.point()),
        AST->getTokens(), AST->getLangOpts());
    if (A.ranges().size()) {
      llvm::StringRef Want = A.code().slice(A.range().Begin, A.range().End);
      EXPECT_EQ(Want, SW->Text) << Text;
    }
    return SW;
  }

protected:
  SpelledWord word(const char *Text) {
    auto Result = tryWord(Text);
    EXPECT_TRUE(Result) << Text;
    return Result.value_or(SpelledWord());
  }

  void noWord(const char *Text) { EXPECT_FALSE(tryWord(Text)) << Text; }
};

TEST_F(SpelledWordsTest, HeuristicBoundaries) {
  word("// [[^foo]] ");
  word("// [[f^oo]] ");
  word("// [[foo^]] ");
  word("// [[foo^]]+bar ");
  noWord("//^ foo ");
  noWord("// foo ^");
}

TEST_F(SpelledWordsTest, LikelyIdentifier) {
  EXPECT_FALSE(word("// ^foo ").LikelyIdentifier);
  EXPECT_TRUE(word("// [[^foo_bar]] ").LikelyIdentifier);
  EXPECT_TRUE(word("// [[^fooBar]] ").LikelyIdentifier);
  EXPECT_FALSE(word("// H^TTP ").LikelyIdentifier);
  EXPECT_TRUE(word("// \\p [[^foo]] ").LikelyIdentifier);
  EXPECT_TRUE(word("// @param[in] [[^foo]] ").LikelyIdentifier);
  EXPECT_TRUE(word("// `[[f^oo]]` ").LikelyIdentifier);
  EXPECT_TRUE(word("// bar::[[f^oo]] ").LikelyIdentifier);
  EXPECT_TRUE(word("// [[f^oo]]::bar ").LikelyIdentifier);
}

TEST_F(SpelledWordsTest, Comment) {
  auto W = word("// [[^foo]]");
  EXPECT_FALSE(W.PartOfSpelledToken);
  EXPECT_FALSE(W.SpelledToken);
  EXPECT_FALSE(W.ExpandedToken);
}

TEST_F(SpelledWordsTest, PartOfString) {
  auto W = word(R"( auto str = "foo [[^bar]] baz"; )");
  ASSERT_TRUE(W.PartOfSpelledToken);
  EXPECT_EQ(W.PartOfSpelledToken->kind(), tok::string_literal);
  EXPECT_FALSE(W.SpelledToken);
  EXPECT_FALSE(W.ExpandedToken);
}

TEST_F(SpelledWordsTest, DisabledSection) {
  auto W = word(R"cpp(
    #if 0
    foo [[^bar]] baz
    #endif
    )cpp");
  ASSERT_TRUE(W.SpelledToken);
  EXPECT_EQ(W.SpelledToken->kind(), tok::identifier);
  EXPECT_EQ(W.SpelledToken, W.PartOfSpelledToken);
  EXPECT_FALSE(W.ExpandedToken);
}

TEST_F(SpelledWordsTest, Macros) {
  auto W = word(R"cpp(
    #define ID(X) X
    ID(int [[^i]]);
    )cpp");
  ASSERT_TRUE(W.SpelledToken);
  EXPECT_EQ(W.SpelledToken->kind(), tok::identifier);
  EXPECT_EQ(W.SpelledToken, W.PartOfSpelledToken);
  ASSERT_TRUE(W.ExpandedToken);
  EXPECT_EQ(W.ExpandedToken->kind(), tok::identifier);

  W = word(R"cpp(
    #define OBJECT Expansion;
    int [[^OBJECT]];
    )cpp");
  EXPECT_TRUE(W.SpelledToken);
  EXPECT_FALSE(W.ExpandedToken) << "Expanded token is spelled differently";
}

TEST(SourceCodeTests, VisibleNamespaces) {
  std::vector<std::pair<const char *, std::vector<std::string>>> Cases = {
      {
          R"cpp(
            // Using directive resolved against enclosing namespaces.
            using namespace foo;
            namespace ns {
            using namespace bar;
          )cpp",
          {"ns", "", "bar", "foo", "ns::bar"},
      },
      {
          R"cpp(
            // Don't include namespaces we've closed, ignore namespace aliases.
            using namespace clang;
            using std::swap;
            namespace clang {
            namespace clangd {}
            namespace ll = ::llvm;
            }
            namespace clang {
          )cpp",
          {"clang", ""},
      },
      {
          R"cpp(
            // Using directives visible even if a namespace is reopened.
            // Ignore anonymous namespaces.
            namespace foo{ using namespace bar; }
            namespace foo{ namespace {
          )cpp",
          {"foo", "", "bar", "foo::bar"},
      },
      {
          R"cpp(
            // Mismatched braces
            namespace foo{}
            }}}
            namespace bar{
          )cpp",
          {"bar", ""},
      },
      {
          R"cpp(
            // Namespaces with multiple chunks.
            namespace a::b {
              using namespace c::d;
              namespace e::f {
          )cpp",
          {
              "a::b::e::f",
              "",
              "a",
              "a::b",
              "a::b::c::d",
              "a::b::e",
              "a::c::d",
              "c::d",
          },
      },
      {
          "",
          {""},
      },
      {
          R"cpp(
            // Parse until EOF
            namespace bar{})cpp",
          {""},
      },
  };
  for (const auto &Case : Cases) {
    EXPECT_EQ(Case.second,
              visibleNamespaces(Case.first, format::getFormattingLangOpts(
                                                format::getLLVMStyle())))
        << Case.first;
  }
}

TEST(SourceCodeTests, GetMacros) {
  Annotations Code(R"cpp(
     #define MACRO 123
     int abc = MA^CRO;
   )cpp");
  TestTU TU = TestTU::withCode(Code.code());
  auto AST = TU.build();
  auto CurLoc = sourceLocationInMainFile(AST.getSourceManager(), Code.point());
  ASSERT_TRUE(bool(CurLoc));
  const auto *Id = syntax::spelledIdentifierTouching(*CurLoc, AST.getTokens());
  ASSERT_TRUE(Id);
  auto Result = locateMacroAt(*Id, AST.getPreprocessor());
  ASSERT_TRUE(Result);
  EXPECT_THAT(*Result, macroName("MACRO"));
}

TEST(SourceCodeTests, WorksAtBeginOfFile) {
  Annotations Code("^MACRO");
  TestTU TU = TestTU::withCode(Code.code());
  TU.HeaderCode = "#define MACRO int x;";
  auto AST = TU.build();
  auto CurLoc = sourceLocationInMainFile(AST.getSourceManager(), Code.point());
  ASSERT_TRUE(bool(CurLoc));
  const auto *Id = syntax::spelledIdentifierTouching(*CurLoc, AST.getTokens());
  ASSERT_TRUE(Id);
  auto Result = locateMacroAt(*Id, AST.getPreprocessor());
  ASSERT_TRUE(Result);
  EXPECT_THAT(*Result, macroName("MACRO"));
}

TEST(SourceCodeTests, IsInsideMainFile) {
  TestTU TU;
  TU.HeaderCode = R"cpp(
    #define DEFINE_CLASS(X) class X {};
    #define DEFINE_YY DEFINE_CLASS(YY)

    class Header1 {};
    DEFINE_CLASS(Header2)
    class Header {};
  )cpp";
  TU.Code = R"cpp(
    #define DEFINE_MAIN4 class Main4{};
    class Main1 {};
    DEFINE_CLASS(Main2)
    DEFINE_YY
    class Main {};
    DEFINE_MAIN4
  )cpp";
  TU.ExtraArgs.push_back("-DHeader=Header3");
  TU.ExtraArgs.push_back("-DMain=Main3");
  auto AST = TU.build();
  const auto &SM = AST.getSourceManager();
  auto DeclLoc = [&AST](llvm::StringRef Name) {
    return findDecl(AST, Name).getLocation();
  };
  for (const auto *HeaderDecl : {"Header1", "Header2", "Header3"})
    EXPECT_FALSE(isInsideMainFile(DeclLoc(HeaderDecl), SM)) << HeaderDecl;

  for (const auto *MainDecl : {"Main1", "Main2", "Main3", "Main4", "YY"})
    EXPECT_TRUE(isInsideMainFile(DeclLoc(MainDecl), SM)) << MainDecl;

  // Main4 is *spelled* in the preamble, but in the main-file part of it.
  EXPECT_TRUE(isInsideMainFile(SM.getSpellingLoc(DeclLoc("Main4")), SM));
}

// Test for functions toHalfOpenFileRange and getHalfOpenFileRange
TEST(SourceCodeTests, HalfOpenFileRange) {
  // Each marked range should be the file range of the decl with the same name
  // and each name should be unique.
  Annotations Test(R"cpp(
    #define FOO(X, Y) int Y = ++X
    #define BAR(X) X + 1
    #define ECHO(X) X

    #define BUZZ BAZZ(ADD)
    #define BAZZ(m) m(1)
    #define ADD(a) int f = a + 1;
    template<typename T>
    class P {};

    int main() {
      $a[[P<P<P<P<P<int>>>>> a]];
      $b[[int b = 1]];
      $c[[FOO(b, c)]]; 
      $d[[FOO(BAR(BAR(b)), d)]];
      // FIXME: We might want to select everything inside the outer ECHO.
      ECHO(ECHO($e[[int) ECHO(e]]));
      // Shouldn't crash.
      $f[[BUZZ]];
    }
  )cpp");

  ParsedAST AST = TestTU::withCode(Test.code()).build();
  llvm::errs() << Test.code();
  const SourceManager &SM = AST.getSourceManager();
  const LangOptions &LangOpts = AST.getLangOpts();
  // Turn a SourceLocation into a pair of positions
  auto SourceRangeToRange = [&SM](SourceRange SrcRange) {
    return Range{sourceLocToPosition(SM, SrcRange.getBegin()),
                 sourceLocToPosition(SM, SrcRange.getEnd())};
  };
  auto CheckRange = [&](llvm::StringRef Name) {
    const NamedDecl &Decl = findUnqualifiedDecl(AST, Name);
    auto FileRange = toHalfOpenFileRange(SM, LangOpts, Decl.getSourceRange());
    SCOPED_TRACE("Checking range: " + Name);
    ASSERT_NE(FileRange, std::nullopt);
    Range HalfOpenRange = SourceRangeToRange(*FileRange);
    EXPECT_EQ(HalfOpenRange, Test.ranges(Name)[0]);
  };

  CheckRange("a");
  CheckRange("b");
  CheckRange("c");
  CheckRange("d");
  CheckRange("e");
  CheckRange("f");
}

TEST(SourceCodeTests, HalfOpenFileRangePathologicalPreprocessor) {
  const char *Case = R"cpp(
#define MACRO while(1)
    void test() {
[[#include "Expand.inc"
        br^eak]];
    }
  )cpp";
  Annotations Test(Case);
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["Expand.inc"] = "MACRO\n";
  auto AST = TU.build();

  const auto &Func = cast<FunctionDecl>(findDecl(AST, "test"));
  const auto &Body = cast<CompoundStmt>(Func.getBody());
  const auto &Loop = cast<WhileStmt>(*Body->child_begin());
  llvm::Optional<SourceRange> Range = toHalfOpenFileRange(
      AST.getSourceManager(), AST.getLangOpts(), Loop->getSourceRange());
  ASSERT_TRUE(Range) << "Failed to get file range";
  EXPECT_EQ(AST.getSourceManager().getFileOffset(Range->getBegin()),
            Test.llvm::Annotations::range().Begin);
  EXPECT_EQ(AST.getSourceManager().getFileOffset(Range->getEnd()),
            Test.llvm::Annotations::range().End);
}

TEST(SourceCodeTests, IncludeHashLoc) {
  const char *Case = R"cpp(
$foo^#include "foo.inc"
#define HEADER "bar.inc"
  $bar^#  include HEADER
  )cpp";
  Annotations Test(Case);
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["foo.inc"] = "int foo;\n";
  TU.AdditionalFiles["bar.inc"] = "int bar;\n";
  auto AST = TU.build();
  const auto &SM = AST.getSourceManager();

  FileID Foo = SM.getFileID(findDecl(AST, "foo").getLocation());
  EXPECT_EQ(SM.getFileOffset(includeHashLoc(Foo, SM)),
            Test.llvm::Annotations::point("foo"));
  FileID Bar = SM.getFileID(findDecl(AST, "bar").getLocation());
  EXPECT_EQ(SM.getFileOffset(includeHashLoc(Bar, SM)),
            Test.llvm::Annotations::point("bar"));
}

TEST(SourceCodeTests, GetEligiblePoints) {
  constexpr struct {
    const char *Code;
    const char *FullyQualifiedName;
    const char *EnclosingNamespace;
  } Cases[] = {
      {R"cpp(// FIXME: We should also mark positions before and after
                 //declarations/definitions as eligible.
              namespace ns1 {
              namespace a { namespace ns2 {} }
              namespace ns2 {^
              void foo();
              namespace {}
              void bar() {}
              namespace ns3 {}
              class T {};
              ^}
              using namespace ns2;
              })cpp",
       "ns1::ns2::symbol", "ns1::ns2::"},
      {R"cpp(
              namespace ns1 {^
              namespace a { namespace ns2 {} }
              namespace b {}
              namespace ns {}
              ^})cpp",
       "ns1::ns2::symbol", "ns1::"},
      {R"cpp(
              namespace x {
              namespace a { namespace ns2 {} }
              namespace b {}
              namespace ns {}
              }^)cpp",
       "ns1::ns2::symbol", ""},
      {R"cpp(
              namespace ns1 {
              namespace ns2 {^^}
              namespace b {}
              namespace ns2 {^^}
              }
              namespace ns1 {namespace ns2 {^^}})cpp",
       "ns1::ns2::symbol", "ns1::ns2::"},
      {R"cpp(
              namespace ns1 {^
              namespace ns {}
              namespace b {}
              namespace ns {}
              ^}
              namespace ns1 {^namespace ns {}^})cpp",
       "ns1::ns2::symbol", "ns1::"},
  };
  for (auto Case : Cases) {
    Annotations Test(Case.Code);

    auto Res = getEligiblePoints(
        Test.code(), Case.FullyQualifiedName,
        format::getFormattingLangOpts(format::getLLVMStyle()));
    EXPECT_THAT(Res.EligiblePoints, testing::ElementsAreArray(Test.points()))
        << Test.code();
    EXPECT_EQ(Res.EnclosingNamespace, Case.EnclosingNamespace) << Test.code();
  }
}

TEST(SourceCodeTests, IdentifierRanges) {
  Annotations Code(R"cpp(
   class [[Foo]] {};
   // Foo
   /* Foo */
   void f([[Foo]]* foo1) {
     [[Foo]] foo2;
     auto S = [[Foo]]();
// cross-line identifier is not supported.
F\
o\
o foo2;
   }
  )cpp");
  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  EXPECT_EQ(Code.ranges(),
            collectIdentifierRanges("Foo", Code.code(), LangOpts));
}

TEST(SourceCodeTests, isHeaderFile) {
  // Without lang options.
  EXPECT_TRUE(isHeaderFile("foo.h"));
  EXPECT_TRUE(isHeaderFile("foo.hh"));
  EXPECT_TRUE(isHeaderFile("foo.hpp"));

  EXPECT_FALSE(isHeaderFile("foo.cpp"));
  EXPECT_FALSE(isHeaderFile("foo.c++"));
  EXPECT_FALSE(isHeaderFile("foo.cxx"));
  EXPECT_FALSE(isHeaderFile("foo.cc"));
  EXPECT_FALSE(isHeaderFile("foo.c"));
  EXPECT_FALSE(isHeaderFile("foo.mm"));
  EXPECT_FALSE(isHeaderFile("foo.m"));

  // With lang options
  LangOptions LangOpts;
  LangOpts.IsHeaderFile = true;
  EXPECT_TRUE(isHeaderFile("string", LangOpts));
  // Emulate cases where there is no "-x header" flag for a .h file, we still
  // want to treat it as a header.
  LangOpts.IsHeaderFile = false;
  EXPECT_TRUE(isHeaderFile("header.h", LangOpts));
}

TEST(SourceCodeTests, isKeywords) {
  LangOptions LangOpts;
  LangOpts.CPlusPlus20 = true;
  EXPECT_TRUE(isKeyword("int", LangOpts));
  EXPECT_TRUE(isKeyword("return", LangOpts));
  EXPECT_TRUE(isKeyword("co_await", LangOpts));

  // these are identifiers (not keywords!) with special meaning in some
  // contexts.
  EXPECT_FALSE(isKeyword("final", LangOpts));
  EXPECT_FALSE(isKeyword("override", LangOpts));
}

struct IncrementalTestStep {
  llvm::StringRef Src;
  llvm::StringRef Contents;
};

int rangeLength(llvm::StringRef Code, const Range &Rng) {
  llvm::Expected<size_t> Start = positionToOffset(Code, Rng.start);
  llvm::Expected<size_t> End = positionToOffset(Code, Rng.end);
  assert(Start);
  assert(End);
  return *End - *Start;
}

/// Send the changes one by one to updateDraft, verify the intermediate results.
void stepByStep(llvm::ArrayRef<IncrementalTestStep> Steps) {
  std::string Code = Annotations(Steps.front().Src).code().str();

  for (size_t I = 1; I < Steps.size(); I++) {
    Annotations SrcBefore(Steps[I - 1].Src);
    Annotations SrcAfter(Steps[I].Src);
    llvm::StringRef Contents = Steps[I - 1].Contents;
    TextDocumentContentChangeEvent Event{
        SrcBefore.range(),
        rangeLength(SrcBefore.code(), SrcBefore.range()),
        Contents.str(),
    };

    EXPECT_THAT_ERROR(applyChange(Code, Event), llvm::Succeeded());
    EXPECT_EQ(Code, SrcAfter.code());
  }
}

TEST(ApplyEditsTest, Simple) {
  // clang-format off
  IncrementalTestStep Steps[] =
    {
      // Replace a range
      {
R"cpp(static int
hello[[World]]()
{})cpp",
        "Universe"
      },
      // Delete a range
      {
R"cpp(static int
hello[[Universe]]()
{})cpp",
        ""
      },
      // Add a range
      {
R"cpp(static int
hello[[]]()
{})cpp",
        "Monde"
      },
      {
R"cpp(static int
helloMonde()
{})cpp",
        ""
      }
    };
  // clang-format on

  stepByStep(Steps);
}

TEST(ApplyEditsTest, MultiLine) {
  // clang-format off
  IncrementalTestStep Steps[] =
    {
      // Replace a range
      {
R"cpp(static [[int
helloWorld]]()
{})cpp",
R"cpp(char
welcome)cpp"
      },
      // Delete a range
      {
R"cpp(static char[[
welcome]]()
{})cpp",
        ""
      },
      // Add a range
      {
R"cpp(static char[[]]()
{})cpp",
        R"cpp(
cookies)cpp"
      },
      // Replace the whole file
      {
R"cpp([[static char
cookies()
{}]])cpp",
        R"cpp(#include <stdio.h>
)cpp"
      },
      // Delete the whole file
      {
        R"cpp([[#include <stdio.h>
]])cpp",
        "",
      },
      // Add something to an empty file
      {
        "[[]]",
        R"cpp(int main() {
)cpp",
      },
      {
        R"cpp(int main() {
)cpp",
        ""
      }
    };
  // clang-format on

  stepByStep(Steps);
}

TEST(ApplyEditsTest, WrongRangeLength) {
  std::string Code = "int main() {}\n";

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 0;
  Change.range->end.character = 2;
  Change.rangeLength = 10;

  EXPECT_THAT_ERROR(applyChange(Code, Change),
                    FailedWithMessage("Change's rangeLength (10) doesn't match "
                                      "the computed range length (2)."));
}

// Test that we correct observed buggy edits from Neovim.
TEST(ApplyEditsTets, BuggyNeovimEdits) {
  TextDocumentContentChangeEvent Change;
  Change.range.emplace();

  // https://github.com/neovim/neovim/issues/17085
  // Adding a blank line after a (missing) newline
  std::string Code = "a";
  Change.range->start.line = 1;
  Change.range->start.character = 0;
  Change.range->end.line = 1;
  Change.range->start.character = 0;
  Change.rangeLength = 0;
  Change.text = "\n";
  EXPECT_THAT_ERROR(applyChange(Code, Change), llvm::Succeeded());
  EXPECT_EQ(Code, "a\n\n");

  // https://github.com/neovim/neovim/issues/17085#issuecomment-1269162264
  // Replacing the (missing) newline with \n\n in an empty file.
  Code = "";
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 1;
  Change.range->end.character = 0;
  Change.rangeLength = 1;
  Change.text = "\n\n";

  EXPECT_THAT_ERROR(applyChange(Code, Change), llvm::Succeeded());
  EXPECT_EQ(Code, "\n\n");

  // We do not apply the heuristic fixes if the rangeLength doesn't match.
  Code = "";
  Change.rangeLength = 0;
  EXPECT_THAT_ERROR(applyChange(Code, Change),
                    FailedWithMessage("Change's rangeLength (0) doesn't match "
                                      "the computed range length (1)."));
}

TEST(ApplyEditsTest, EndBeforeStart) {
  std::string Code = "int main() {}\n";

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 5;
  Change.range->end.line = 0;
  Change.range->end.character = 3;

  EXPECT_THAT_ERROR(
      applyChange(Code, Change),
      FailedWithMessage(
          "Range's end position (0:3) is before start position (0:5)"));
}

TEST(ApplyEditsTest, StartCharOutOfRange) {
  std::string Code = "int main() {}\n";

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 100;
  Change.range->end.line = 0;
  Change.range->end.character = 100;
  Change.text = "foo";

  EXPECT_THAT_ERROR(
      applyChange(Code, Change),
      FailedWithMessage("utf-16 offset 100 is invalid for line 0"));
}

TEST(ApplyEditsTest, EndCharOutOfRange) {
  std::string Code = "int main() {}\n";

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 0;
  Change.range->end.character = 100;
  Change.text = "foo";

  EXPECT_THAT_ERROR(
      applyChange(Code, Change),
      FailedWithMessage("utf-16 offset 100 is invalid for line 0"));
}

TEST(ApplyEditsTest, StartLineOutOfRange) {
  std::string Code = "int main() {}\n";

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 100;
  Change.range->start.character = 0;
  Change.range->end.line = 100;
  Change.range->end.character = 0;
  Change.text = "foo";

  EXPECT_THAT_ERROR(applyChange(Code, Change),
                    FailedWithMessage("Line value is out of range (100)"));
}

TEST(ApplyEditsTest, EndLineOutOfRange) {
  std::string Code = "int main() {}\n";

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 100;
  Change.range->end.character = 0;
  Change.text = "foo";

  EXPECT_THAT_ERROR(applyChange(Code, Change),
                    FailedWithMessage("Line value is out of range (100)"));
}

} // namespace
} // namespace clangd
} // namespace clang
