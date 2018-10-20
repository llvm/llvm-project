//===-- GlobalCompilationDatabaseTests.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"

#include "TestFS.h"
#include "llvm/ADT/StringExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace {
using ::testing::ElementsAre;

TEST(GlobalCompilationDatabaseTest, FallbackCommand) {
  DirectoryBasedGlobalCompilationDatabase DB(None);
  auto Cmd = DB.getFallbackCommand(testPath("foo/bar.cc"));
  EXPECT_EQ(Cmd.Directory, testPath("foo"));
  EXPECT_THAT(Cmd.CommandLine, ElementsAre("clang", testPath("foo/bar.cc")));
  EXPECT_EQ(Cmd.Output, "");

  // .h files have unknown language, so they are parsed liberally as obj-c++.
  Cmd = DB.getFallbackCommand(testPath("foo/bar.h"));
  EXPECT_THAT(Cmd.CommandLine, ElementsAre("clang", "-xobjective-c++-header",
                                           testPath("foo/bar.h")));
}

} // namespace
} // namespace clangd
} // namespace clang
