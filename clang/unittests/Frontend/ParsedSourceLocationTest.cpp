//===- unittests/Frontend/ParsedSourceLocationTest.cpp - ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CommandLineSourceLoc.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

TEST(ParsedSourceRange, ParseTest) {
  auto Check = [](StringRef Value, StringRef Filename, unsigned BeginLine,
                  unsigned BeginColumn, unsigned EndLine, unsigned EndColumn) {
    Optional<ParsedSourceRange> PSR = ParsedSourceRange::fromString(Value);
    ASSERT_TRUE(PSR);
    EXPECT_EQ(PSR->FileName, Filename);
    EXPECT_EQ(PSR->Begin.first, BeginLine);
    EXPECT_EQ(PSR->Begin.second, BeginColumn);
    EXPECT_EQ(PSR->End.first, EndLine);
    EXPECT_EQ(PSR->End.second, EndColumn);
  };

  Check("/Users/test/a-b.cpp:1:2", "/Users/test/a-b.cpp", 1, 2, 1, 2);
  Check("/Users/test/a-b.cpp:1:2-3:4", "/Users/test/a-b.cpp", 1, 2, 3, 4);

  Check("C:/Users/bob/a-b.cpp:1:2", "C:/Users/bob/a-b.cpp", 1, 2, 1, 2);
  Check("C:/Users/bob/a-b.cpp:1:2-3:4", "C:/Users/bob/a-b.cpp", 1, 2, 3, 4);
}

} // anonymous namespace
