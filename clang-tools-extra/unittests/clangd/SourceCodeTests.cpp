//===-- SourceCodeTests.cpp  ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "SourceCode.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang{
namespace clangd {
namespace {

using llvm::Failed;
using llvm::HasValue;

MATCHER_P2(Pos, Line, Col, "") {
  return arg.line == Line && arg.character == Col;
}

const char File[] = R"(0:0 = 0
1:0 = 8
2:0 = 16)";

/// A helper to make tests easier to read.
Position position(int line, int character) {
  Position Pos;
  Pos.line = line;
  Pos.character = character;
  return Pos;
}

TEST(SourceCodeTests, PositionToOffset) {
  // line out of bounds
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(-1, 2)), Failed());
  // first line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, -1)),
                       Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 0)),
                       HasValue(0)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 3)),
                       HasValue(3)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 6)),
                       HasValue(6)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 7)),
                       HasValue(7)); // the newline itself
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 7), false),
                       HasValue(7));
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 8)),
                       HasValue(7)); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(0, 8), false),
                       Failed()); // out of range
  // middle line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, -1)),
                       Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 0)),
                       HasValue(8)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 3)),
                       HasValue(11)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 3), false),
                       HasValue(11));
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 6)),
                       HasValue(14)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 7)),
                       HasValue(15)); // the newline itself
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 8)),
                       HasValue(15)); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(1, 8), false),
                       Failed()); // out of range
  // last line
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, -1)),
                       Failed()); // out of range
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 0)),
                       HasValue(16)); // first character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 3)),
                       HasValue(19)); // middle character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 7)),
                       HasValue(23)); // last character
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 8)),
                       HasValue(24)); // EOF
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(2, 9), false),
                       Failed()); // out of range
  // line out of bounds
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 0)), Failed());
  EXPECT_THAT_EXPECTED(positionToOffset(File, position(3, 1)), Failed());
}

TEST(SourceCodeTests, OffsetToPosition) {
  EXPECT_THAT(offsetToPosition(File, 0), Pos(0, 0)) << "start of file";
  EXPECT_THAT(offsetToPosition(File, 3), Pos(0, 3)) << "in first line";
  EXPECT_THAT(offsetToPosition(File, 6), Pos(0, 6)) << "end of first line";
  EXPECT_THAT(offsetToPosition(File, 7), Pos(0, 7)) << "first newline";
  EXPECT_THAT(offsetToPosition(File, 8), Pos(1, 0)) << "start of second line";
  EXPECT_THAT(offsetToPosition(File, 11), Pos(1, 3)) << "in second line";
  EXPECT_THAT(offsetToPosition(File, 14), Pos(1, 6)) << "end of second line";
  EXPECT_THAT(offsetToPosition(File, 15), Pos(1, 7)) << "second newline";
  EXPECT_THAT(offsetToPosition(File, 16), Pos(2, 0)) << "start of last line";
  EXPECT_THAT(offsetToPosition(File, 19), Pos(2, 3)) << "in last line";
  EXPECT_THAT(offsetToPosition(File, 23), Pos(2, 7)) << "end of last line";
  EXPECT_THAT(offsetToPosition(File, 24), Pos(2, 8)) << "EOF";
  EXPECT_THAT(offsetToPosition(File, 25), Pos(2, 8)) << "out of bounds";
}

} // namespace
} // namespace clangd
} // namespace clang
