//===-- ConfigCompileTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "ConfigFragment.h"
#include "ConfigTesting.h"
#include "Features.inc"
#include "TestFS.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

namespace clang {
namespace clangd {
namespace config {
namespace {
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StartsWith;

class ConfigCompileTests : public ::testing::Test {
protected:
  CapturedDiags Diags;
  Config Conf;
  Fragment Frag;
  Params Parm;

  bool compileAndApply() {
    Conf = Config();
    Diags.Diagnostics.clear();
    auto Compiled = std::move(Frag).compile(Diags.callback());
    return Compiled(Parm, Conf);
  }
};

TEST_F(ConfigCompileTests, Condition) {
  // No condition.
  Frag = {};
  Frag.CompileFlags.Add.emplace_back("X");
  EXPECT_TRUE(compileAndApply()) << "Empty config";
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(Conf.CompileFlags.Edits, SizeIs(1));

  // Regex with no file.
  Frag = {};
  Frag.If.PathMatch.emplace_back("fo*");
  EXPECT_FALSE(compileAndApply());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(Conf.CompileFlags.Edits, SizeIs(0));

  // Following tests have a file path set.
  Parm.Path = "bar";

  // Non-matching regex.
  Frag = {};
  Frag.If.PathMatch.emplace_back("fo*");
  EXPECT_FALSE(compileAndApply());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());

  // Matching regex.
  Frag = {};
  Frag.If.PathMatch.emplace_back("fo*");
  Frag.If.PathMatch.emplace_back("ba*r");
  EXPECT_TRUE(compileAndApply());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());

  // Excluded regex.
  Frag = {};
  Frag.If.PathMatch.emplace_back("b.*");
  Frag.If.PathExclude.emplace_back(".*r");
  EXPECT_FALSE(compileAndApply()) << "Included but also excluded";
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());

  // Invalid regex.
  Frag = {};
  Frag.If.PathMatch.emplace_back("**]@theu");
  EXPECT_TRUE(compileAndApply());
  EXPECT_THAT(Diags.Diagnostics, SizeIs(1));
  EXPECT_THAT(Diags.Diagnostics.front().Message, StartsWith("Invalid regex"));

  // Valid regex and unknown key.
  Frag = {};
  Frag.If.HasUnrecognizedCondition = true;
  Frag.If.PathMatch.emplace_back("ba*r");
  EXPECT_FALSE(compileAndApply());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
}

TEST_F(ConfigCompileTests, CompileCommands) {
  Frag.CompileFlags.Add.emplace_back("-foo");
  Frag.CompileFlags.Remove.emplace_back("--include-directory=");
  std::vector<std::string> Argv = {"clang", "-I", "bar/", "a.cc"};
  EXPECT_TRUE(compileAndApply());
  EXPECT_THAT(Conf.CompileFlags.Edits, SizeIs(2));
  for (auto &Edit : Conf.CompileFlags.Edits)
    Edit(Argv);
  EXPECT_THAT(Argv, ElementsAre("clang", "a.cc", "-foo"));
}

TEST_F(ConfigCompileTests, Index) {
  Frag.Index.Background.emplace("Skip");
  EXPECT_TRUE(compileAndApply());
  EXPECT_EQ(Conf.Index.Background, Config::BackgroundPolicy::Skip);

  Frag = {};
  Frag.Index.Background.emplace("Foo");
  EXPECT_TRUE(compileAndApply());
  EXPECT_EQ(Conf.Index.Background, Config::BackgroundPolicy::Build)
      << "by default";
  EXPECT_THAT(
      Diags.Diagnostics,
      ElementsAre(DiagMessage(
          "Invalid Background value 'Foo'. Valid values are Build, Skip.")));
}

TEST_F(ConfigCompileTests, PathSpecMatch) {
  auto BarPath = llvm::sys::path::convert_to_slash(testPath("foo/bar.h"));
  Parm.Path = BarPath;

  struct {
    std::string Directory;
    std::string PathSpec;
    bool ShouldMatch;
  } Cases[] = {
      {
          // Absolute path matches.
          "",
          llvm::sys::path::convert_to_slash(testPath("foo/bar.h")),
          true,
      },
      {
          // Absolute path fails.
          "",
          llvm::sys::path::convert_to_slash(testPath("bar/bar.h")),
          false,
      },
      {
          // Relative should fail to match as /foo/bar.h doesn't reside under
          // /baz/.
          testPath("baz"),
          "bar\\.h",
          false,
      },
      {
          // Relative should pass with /foo as directory.
          testPath("foo"),
          "bar\\.h",
          true,
      },
  };

  // PathMatch
  for (const auto &Case : Cases) {
    Frag = {};
    Frag.If.PathMatch.emplace_back(Case.PathSpec);
    Frag.Source.Directory = Case.Directory;
    EXPECT_EQ(compileAndApply(), Case.ShouldMatch);
    ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  }

  // PathEclude
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Directory);
    SCOPED_TRACE(Case.PathSpec);
    Frag = {};
    Frag.If.PathExclude.emplace_back(Case.PathSpec);
    Frag.Source.Directory = Case.Directory;
    EXPECT_NE(compileAndApply(), Case.ShouldMatch);
    ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  }
}

TEST_F(ConfigCompileTests, Tidy) {
  Frag.ClangTidy.Add.emplace_back("bugprone-use-after-move");
  Frag.ClangTidy.Add.emplace_back("llvm-*");
  Frag.ClangTidy.Remove.emplace_back("llvm-include-order");
  Frag.ClangTidy.Remove.emplace_back("readability-*");
  Frag.ClangTidy.CheckOptions.emplace_back(
      std::make_pair(std::string("StrictMode"), std::string("true")));
  Frag.ClangTidy.CheckOptions.emplace_back(std::make_pair(
      std::string("example-check.ExampleOption"), std::string("0")));
  EXPECT_TRUE(compileAndApply());
  EXPECT_EQ(
      Conf.ClangTidy.Checks,
      "bugprone-use-after-move,llvm-*,-llvm-include-order,-readability-*");
  EXPECT_EQ(Conf.ClangTidy.CheckOptions.size(), 2U);
  EXPECT_EQ(Conf.ClangTidy.CheckOptions.lookup("StrictMode"), "true");
  EXPECT_EQ(Conf.ClangTidy.CheckOptions.lookup("example-check.ExampleOption"),
            "0");
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
}

TEST_F(ConfigCompileTests, TidyBadChecks) {
  Frag.ClangTidy.Add.emplace_back("unknown-check");
  Frag.ClangTidy.Remove.emplace_back("*");
  Frag.ClangTidy.Remove.emplace_back("llvm-includeorder");
  EXPECT_TRUE(compileAndApply());
  // Ensure bad checks are stripped from the glob.
  EXPECT_EQ(Conf.ClangTidy.Checks, "-*");
  EXPECT_THAT(
      Diags.Diagnostics,
      ElementsAre(
          AllOf(DiagMessage("clang-tidy check 'unknown-check' was not found"),
                DiagKind(llvm::SourceMgr::DK_Warning)),
          AllOf(
              DiagMessage("clang-tidy check 'llvm-includeorder' was not found"),
              DiagKind(llvm::SourceMgr::DK_Warning))));
}

TEST_F(ConfigCompileTests, ExternalBlockWarnOnMultipleSource) {
  Fragment::IndexBlock::ExternalBlock External;
  External.File.emplace("");
  External.Server.emplace("");
  Frag.Index.External = std::move(External);
  compileAndApply();
  llvm::StringLiteral ExpectedDiag =
#ifdef CLANGD_ENABLE_REMOTE
      "Exactly one of File or Server must be set.";
#else
      "Clangd isn't compiled with remote index support, ignoring Server.";
#endif
  EXPECT_THAT(Diags.Diagnostics,
              Contains(AllOf(DiagMessage(ExpectedDiag),
                             DiagKind(llvm::SourceMgr::DK_Error))));
}

TEST_F(ConfigCompileTests, ExternalBlockErrOnNoSource) {
  Frag.Index.External.emplace(Fragment::IndexBlock::ExternalBlock{});
  compileAndApply();
  EXPECT_THAT(
      Diags.Diagnostics,
      Contains(AllOf(DiagMessage("Exactly one of File or Server must be set."),
                     DiagKind(llvm::SourceMgr::DK_Error))));
}

TEST_F(ConfigCompileTests, ExternalBlockDisablesBackgroundIndex) {
  auto BazPath = testPath("foo/bar/baz.h", llvm::sys::path::Style::posix);
  Parm.Path = BazPath;
  Frag.Index.Background.emplace("Build");
  Fragment::IndexBlock::ExternalBlock External;
  External.File.emplace(testPath("foo"));
  External.MountPoint.emplace(
      testPath("foo/bar", llvm::sys::path::Style::posix));
  Frag.Index.External = std::move(External);
  compileAndApply();
  EXPECT_EQ(Conf.Index.Background, Config::BackgroundPolicy::Skip);
}

TEST_F(ConfigCompileTests, ExternalBlockMountPoint) {
  auto GetFrag = [](llvm::StringRef Directory,
                    llvm::Optional<const char *> MountPoint) {
    Fragment Frag;
    Frag.Source.Directory = Directory.str();
    Fragment::IndexBlock::ExternalBlock External;
    External.File.emplace(testPath("foo"));
    if (MountPoint)
      External.MountPoint.emplace(*MountPoint);
    Frag.Index.External = std::move(External);
    return Frag;
  };

  auto BarPath = testPath("foo/bar.h", llvm::sys::path::Style::posix);
  BarPath = llvm::sys::path::convert_to_slash(BarPath);
  Parm.Path = BarPath;
  // Non-absolute MountPoint without a directory raises an error.
  Frag = GetFrag("", "foo");
  compileAndApply();
  ASSERT_THAT(
      Diags.Diagnostics,
      ElementsAre(
          AllOf(DiagMessage("MountPoint must be an absolute path, because this "
                            "fragment is not associated with any directory."),
                DiagKind(llvm::SourceMgr::DK_Error))));
  ASSERT_FALSE(Conf.Index.External);

  auto FooPath = testPath("foo/", llvm::sys::path::Style::posix);
  FooPath = llvm::sys::path::convert_to_slash(FooPath);
  // Ok when relative.
  Frag = GetFrag(testRoot(), "foo/");
  compileAndApply();
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_TRUE(Conf.Index.External);
  EXPECT_THAT(Conf.Index.External->MountPoint, FooPath);

  // None defaults to ".".
  Frag = GetFrag(FooPath, llvm::None);
  compileAndApply();
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_TRUE(Conf.Index.External);
  EXPECT_THAT(Conf.Index.External->MountPoint, FooPath);

  // Without a file, external index is empty.
  Parm.Path = "";
  Frag = GetFrag("", FooPath.c_str());
  compileAndApply();
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_FALSE(Conf.Index.External);

  // File outside MountPoint, no index.
  auto BazPath = testPath("bar/baz.h", llvm::sys::path::Style::posix);
  BazPath = llvm::sys::path::convert_to_slash(BazPath);
  Parm.Path = BazPath;
  Frag = GetFrag("", FooPath.c_str());
  compileAndApply();
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_FALSE(Conf.Index.External);

  // File under MountPoint, index should be set.
  BazPath = testPath("foo/baz.h", llvm::sys::path::Style::posix);
  BazPath = llvm::sys::path::convert_to_slash(BazPath);
  Parm.Path = BazPath;
  Frag = GetFrag("", FooPath.c_str());
  compileAndApply();
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_TRUE(Conf.Index.External);
  EXPECT_THAT(Conf.Index.External->MountPoint, FooPath);
}
} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
