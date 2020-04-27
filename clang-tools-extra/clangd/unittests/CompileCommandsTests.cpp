//===-- CompileCommandsTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "TestFS.h"

#include "llvm/ADT/StringExtras.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;

// Sadly, CommandMangler::detect(), which contains much of the logic, is
// a bunch of untested integration glue. We test the string manipulation here
// assuming its results are correct.

// Make use of all features and assert the exact command we get out.
// Other tests just verify presence/absence of certain args.
TEST(CommandMangler, Everything) {
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");
  Mangler.ResourceDir = testPath("fake/resources");
  Mangler.Sysroot = testPath("fake/sysroot");
  std::vector<std::string> Cmd = {"clang++", "-Xclang", "-load", "-Xclang",
                                  "plugin",  "-MF",     "dep",   "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_THAT(Cmd, ElementsAre(testPath("fake/clang++"), "foo.cc",
                               "-fsyntax-only",
                               "-resource-dir=" + testPath("fake/resources"),
                               "-isysroot", testPath("fake/sysroot")));
}

TEST(CommandMangler, ResourceDir) {
  auto Mangler = CommandMangler::forTests();
  Mangler.ResourceDir = testPath("fake/resources");
  std::vector<std::string> Cmd = {"clang++", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_THAT(Cmd, Contains("-resource-dir=" + testPath("fake/resources")));
}

TEST(CommandMangler, Sysroot) {
  auto Mangler = CommandMangler::forTests();
  Mangler.Sysroot = testPath("fake/sysroot");

  std::vector<std::string> Cmd = {"clang++", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_THAT(llvm::join(Cmd, " "),
              HasSubstr("-isysroot " + testPath("fake/sysroot")));
}

TEST(CommandMangler, StripPlugins) {
  auto Mangler = CommandMangler::forTests();
  std::vector<std::string> Cmd = {"clang++", "-Xclang", "-load",
                                  "-Xclang", "plugin",  "foo.cc"};
  Mangler.adjust(Cmd);
  for (const char* Stripped : {"-Xclang", "-load", "plugin"})
    EXPECT_THAT(Cmd, Not(Contains(Stripped)));
}

TEST(CommandMangler, StripOutput) {
  auto Mangler = CommandMangler::forTests();
  std::vector<std::string> Cmd = {"clang++", "-MF", "dependency", "-c",
                                  "foo.cc"};
  Mangler.adjust(Cmd);
  for (const char* Stripped : {"-MF", "dependency"})
    EXPECT_THAT(Cmd, Not(Contains(Stripped)));
}

TEST(CommandMangler, StripShowIncludes) {
  auto Mangler = CommandMangler::forTests();
  std::vector<std::string> Cmd = {"clang-cl", "/showIncludes", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_THAT(Cmd, Not(Contains("/showIncludes")));
}

TEST(CommandMangler, StripShowIncludesUser) {
  auto Mangler = CommandMangler::forTests();
  std::vector<std::string> Cmd = {"clang-cl", "/showIncludes:user", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_THAT(Cmd, Not(Contains("/showIncludes:user")));
}

TEST(CommandMangler, ClangPath) {
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");

  std::vector<std::string> Cmd = {"clang++", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_EQ(testPath("fake/clang++"), Cmd.front());

  Cmd = {"unknown-binary", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_EQ("unknown-binary", Cmd.front());

  Cmd = {testPath("path/clang++"), "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_EQ(testPath("path/clang++"), Cmd.front());
}

} // namespace
} // namespace clangd
} // namespace clang

