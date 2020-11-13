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
#include "TestFS.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

namespace clang {
namespace clangd {
namespace config {
namespace {
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
} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
