//===-- CompileCommandsTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "TestFS.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

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
  EXPECT_EQ(testPath("fake/unknown-binary"), Cmd.front());

  Cmd = {testPath("path/clang++"), "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_EQ(testPath("path/clang++"), Cmd.front());

  Cmd = {"foo/unknown-binary", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_EQ(testPath("fake/unknown-binary"), Cmd.front());
}

// Only run the PATH/symlink resolving test on unix, we need to fiddle
// with permissions and environment variables...
#ifdef LLVM_ON_UNIX
MATCHER(Ok, "") {
  if (arg) {
    *result_listener << arg.message();
    return false;
  }
  return true;
}

TEST(CommandMangler, ClangPathResolve) {
  // Set up filesystem:
  //   /temp/
  //     bin/
  //       foo -> temp/lib/bar
  //     lib/
  //       bar
  llvm::SmallString<256> TempDir;
  ASSERT_THAT(llvm::sys::fs::createUniqueDirectory("ClangPathResolve", TempDir),
              Ok());
  // /var/tmp is a symlink on Mac. Resolve it so we're asserting the right path.
  ASSERT_THAT(llvm::sys::fs::real_path(TempDir.str(), TempDir), Ok());
  auto CleanDir = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(TempDir); });
  ASSERT_THAT(llvm::sys::fs::create_directory(TempDir + "/bin"), Ok());
  ASSERT_THAT(llvm::sys::fs::create_directory(TempDir + "/lib"), Ok());
  int FD;
  ASSERT_THAT(llvm::sys::fs::openFileForWrite(TempDir + "/lib/bar", FD), Ok());
  ASSERT_THAT(llvm::sys::Process::SafelyCloseFileDescriptor(FD), Ok());
  ::chmod((TempDir + "/lib/bar").str().c_str(), 0755); // executable
  ASSERT_THAT(
      llvm::sys::fs::create_link(TempDir + "/lib/bar", TempDir + "/bin/foo"),
      Ok());

  // Test the case where the driver is an absolute path to a symlink.
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");
  std::vector<std::string> Cmd = {(TempDir + "/bin/foo").str(), "foo.cc"};
  Mangler.adjust(Cmd);
  // Directory based on resolved symlink, basename preserved.
  EXPECT_EQ((TempDir + "/lib/foo").str(), Cmd.front());

  // Set PATH to point to temp/bin so we can find 'foo' on it.
  ASSERT_TRUE(::getenv("PATH"));
  auto RestorePath =
      llvm::make_scope_exit([OldPath = std::string(::getenv("PATH"))] {
        ::setenv("PATH", OldPath.c_str(), 1);
      });
  ::setenv("PATH", (TempDir + "/bin").str().c_str(), /*overwrite=*/1);

  // Test the case where the driver is a $PATH-relative path to a symlink.
  Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");
  // Driver found on PATH.
  Cmd = {"foo", "foo.cc"};
  Mangler.adjust(Cmd);
  // Found the symlink and resolved the path as above.
  EXPECT_EQ((TempDir + "/lib/foo").str(), Cmd.front());

  // Symlink not resolved with -no-canonical-prefixes.
  Cmd = {"foo", "-no-canonical-prefixes", "foo.cc"};
  Mangler.adjust(Cmd);
  EXPECT_EQ((TempDir + "/bin/foo").str(), Cmd.front());
}
#endif

} // namespace
} // namespace clangd
} // namespace clang

