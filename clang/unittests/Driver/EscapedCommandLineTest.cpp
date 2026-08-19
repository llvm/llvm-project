//===- unittests/Driver/EscapedCommandLineTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for -record-command-line.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/CommonArgs.h"
#include "clang/Options/OptionUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/iterator_range.h"
#include "gtest/gtest.h"

using namespace clang::driver::tools;
using namespace llvm;

using ArgStr = SmallString<8>;
using ArgVec = SmallVector<ArgStr>;

static ArgStr escape(const char *Arg) {
  ArgStr Res;
  escapeSpacesAndBackslashes(Arg, Res);
  return Res;
}

TEST(EscapedCommandLineTest, EscapeEmpty) { EXPECT_EQ(escape(""), ""); }

TEST(EscapedCommandLineTest, EscapeNoSpecialChars) {
  EXPECT_EQ(escape("hello"), "hello");
  EXPECT_EQ(escape("-Tlib_6_3"), "-Tlib_6_3");
}

TEST(EscapedCommandLineTest, EscapeSpace) {
  EXPECT_EQ(escape("foo bar"), "foo\\ bar");
  EXPECT_EQ(escape(" leading"), "\\ leading");
  EXPECT_EQ(escape("trailing "), "trailing\\ ");
}

TEST(EscapedCommandLineTest, EscapeBackslash) {
  EXPECT_EQ(escape("a\\b"), "a\\\\b");
}

TEST(EscapedCommandLineTest, EscapeSpaceAndBackslash) {
  EXPECT_EQ(escape("a\\ b"), "a\\\\\\ b");
}

static ArgVec parse(const char *CommandLine) {
  ArgVec Res;
  auto ParsedArgs = clang::parseEscapedCommandLine(CommandLine);
  if (!ParsedArgs) {
    ADD_FAILURE() << llvm::toString(ParsedArgs.takeError());
    return Res;
  }
  for (const auto &Arg : *ParsedArgs)
    Res.emplace_back(Arg.begin(), Arg.end());
  return Res;
}

TEST(EscapedCommandLineTest, ParseEmpty) { EXPECT_TRUE(parse("").empty()); }

TEST(EscapedCommandLineTest, ParseSingleArg) {
  EXPECT_EQ(parse("hello"), ArgVec({StringRef("hello")}));
}

TEST(EscapedCommandLineTest, ParseMultipleArgs) {
  auto Args = parse("clang -Tlib_6_3 foo.hlsl");
  ASSERT_EQ(Args.size(), 3u);
  EXPECT_EQ(Args[0], "clang");
  EXPECT_EQ(Args[1], "-Tlib_6_3");
  EXPECT_EQ(Args[2], "foo.hlsl");
}

TEST(EscapedCommandLineTest, ParseEscapedSpace) {
  auto Args = parse("foo\\ bar baz");
  ASSERT_EQ(Args.size(), 2u);
  EXPECT_EQ(Args[0], "foo bar");
  EXPECT_EQ(Args[1], "baz");
}

TEST(EscapedCommandLineTest, ParseEscapedBackslash) {
  auto Args = parse("a\\\\b");
  ASSERT_EQ(Args.size(), 1u);
  EXPECT_EQ(Args[0], "a\\b");
}

TEST(EscapedCommandLineTest, ParseInvalidEscape) {
  auto Args = clang::parseEscapedCommandLine("clang \\");
  ASSERT_FALSE(Args);
  EXPECT_EQ(llvm::toString(Args.takeError()),
            "only escaped backslashes and spaces are supported");
}

static ArgVec roundTrip(ArgVec Args) {
  SmallString<256> Joined;
  escapeSpacesAndBackslashes(Args.begin()->c_str(), Joined);
  for (auto &Arg : llvm::make_range(Args.begin() + 1, Args.end())) {
    Joined += " ";
    escapeSpacesAndBackslashes(Arg.c_str(), Joined);
  }
  return parse(Joined.c_str());
}

TEST(EscapedCommandLineTest, RoundTripSimple) {
  ArgVec Args;
  Args.emplace_back("clang");
  Args.emplace_back("-O2");
  Args.emplace_back("foo.cpp");
  EXPECT_EQ(roundTrip(Args), Args);
}

TEST(EscapedCommandLineTest, RoundTripArgWithSpace) {
  ArgVec Args;
  Args.emplace_back("clang");
  Args.emplace_back("path with spaces/file.cpp");
  EXPECT_EQ(roundTrip(Args), Args);
}

TEST(EscapedCommandLineTest, RoundTripArgWithBackslash) {
  ArgVec Args;
  Args.emplace_back("clang");
  Args.emplace_back("C:\\path\\file.cpp");
  EXPECT_EQ(roundTrip(Args), Args);
}

TEST(EscapedCommandLineTest, RoundTripArgWithSpaceAndBackslash) {
  ArgVec Args;
  Args.emplace_back("clang");
  Args.emplace_back("C:\\path with space\\file.cpp");
  EXPECT_EQ(roundTrip(Args), Args);
}
