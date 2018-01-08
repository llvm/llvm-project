//===-- SourceCodeTests.cpp  ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "SourceCode.h"
#include "llvm/Support/raw_os_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang{
namespace clangd {
namespace {

MATCHER_P2(Pos, Line, Col, "") {
  return arg.line == Line && arg.character == Col;
}

const char File[] = R"(0:0 = 0
1:0 = 8
2:0 = 16)";

TEST(SourceCodeTests, PositionToOffset) {
  // line out of bounds
  EXPECT_EQ(0u, positionToOffset(File, Position{-1, 2}));
  // first line
  EXPECT_EQ(0u, positionToOffset(File, Position{0, -1})); // out of range
  EXPECT_EQ(0u, positionToOffset(File, Position{0, 0})); // first character
  EXPECT_EQ(3u, positionToOffset(File, Position{0, 3})); // middle character
  EXPECT_EQ(6u, positionToOffset(File, Position{0, 6})); // last character
  EXPECT_EQ(7u, positionToOffset(File, Position{0, 7})); // the newline itself
  EXPECT_EQ(8u, positionToOffset(File, Position{0, 8})); // out of range
  // middle line
  EXPECT_EQ(8u, positionToOffset(File, Position{1, -1})); // out of range
  EXPECT_EQ(8u, positionToOffset(File, Position{1, 0})); // first character
  EXPECT_EQ(11u, positionToOffset(File, Position{1, 3})); // middle character
  EXPECT_EQ(14u, positionToOffset(File, Position{1, 6})); // last character
  EXPECT_EQ(15u, positionToOffset(File, Position{1, 7})); // the newline itself
  EXPECT_EQ(16u, positionToOffset(File, Position{1, 8})); // out of range
  // last line
  EXPECT_EQ(16u, positionToOffset(File, Position{2, -1})); // out of range
  EXPECT_EQ(16u, positionToOffset(File, Position{2, 0})); // first character
  EXPECT_EQ(19u, positionToOffset(File, Position{2, 3})); // middle character
  EXPECT_EQ(23u, positionToOffset(File, Position{2, 7})); // last character
  EXPECT_EQ(24u, positionToOffset(File, Position{2, 8})); // EOF
  EXPECT_EQ(24u, positionToOffset(File, Position{2, 9})); // out of range
  // line out of bounds
  EXPECT_EQ(24u, positionToOffset(File, Position{3, 1}));
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
