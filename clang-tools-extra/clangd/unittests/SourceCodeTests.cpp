//===-- SourceCodeTests.cpp  ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "Context.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using llvm::Failed;
using llvm::HasValue;

MATCHER_P2(Pos, Line, Col, "") {
  return arg.line == int(Line) && arg.character == int(Col);
}

/// A helper to make tests easier to read.
Position position(int line, int character) {
  Position Pos;
  Pos.line = line;
  Pos.character = character;
  return Pos;
}

Range range(const std::pair<int, int> p1, const std::pair<int, int> p2) {
  Range range;
  range.start = position(p1.first, p1.second);
  range.end = position(p2.first, p2.second);
  return range;
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
    EXPECT_THAT_EXPECTED(positionToOffset(File, position(L.Number, L.Length+1)),
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

TEST(SourceCodeTests, IsRangeConsecutive) {
  EXPECT_TRUE(isRangeConsecutive(range({2, 2}, {2, 3}), range({2, 3}, {2, 4})));
  EXPECT_FALSE(
      isRangeConsecutive(range({0, 2}, {0, 3}), range({2, 3}, {2, 4})));
  EXPECT_FALSE(
      isRangeConsecutive(range({2, 2}, {2, 3}), range({2, 4}, {2, 5})));
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
  };
  for (const auto& Case : Cases) {
    EXPECT_EQ(Case.second,
              visibleNamespaces(Case.first, format::getLLVMStyle()))
        << Case.first;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
