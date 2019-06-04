//===-- GlobalCompilationDatabaseTests.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"

#include "TestFS.h"
#include "llvm/ADT/StringExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::EndsWith;
using ::testing::Not;
using ::testing::StartsWith;

TEST(GlobalCompilationDatabaseTest, FallbackCommand) {
  DirectoryBasedGlobalCompilationDatabase DB(None);
  auto Cmd = DB.getFallbackCommand(testPath("foo/bar.cc"));
  EXPECT_EQ(Cmd.Directory, testPath("foo"));
  EXPECT_THAT(Cmd.CommandLine,
              ElementsAre(EndsWith("clang"), testPath("foo/bar.cc")));
  EXPECT_EQ(Cmd.Output, "");

  // .h files have unknown language, so they are parsed liberally as obj-c++.
  Cmd = DB.getFallbackCommand(testPath("foo/bar.h"));
  EXPECT_THAT(Cmd.CommandLine,
              ElementsAre(EndsWith("clang"), "-xobjective-c++-header",
                          testPath("foo/bar.h")));
}

static tooling::CompileCommand cmd(llvm::StringRef File, llvm::StringRef Arg) {
  return tooling::CompileCommand(testRoot(), File, {"clang", Arg, File}, "");
}

class OverlayCDBTest : public ::testing::Test {
  class BaseCDB : public GlobalCompilationDatabase {
  public:
    llvm::Optional<tooling::CompileCommand>
    getCompileCommand(llvm::StringRef File,
                      ProjectInfo *Project) const override {
      if (File == testPath("foo.cc")) {
        if (Project)
          Project->SourceRoot = testRoot();
        return cmd(File, "-DA=1");
      }
      return None;
    }

    tooling::CompileCommand
    getFallbackCommand(llvm::StringRef File) const override {
      return cmd(File, "-DA=2");
    }
  };

protected:
  OverlayCDBTest() : Base(llvm::make_unique<BaseCDB>()) {}
  std::unique_ptr<GlobalCompilationDatabase> Base;
};

TEST_F(OverlayCDBTest, GetCompileCommand) {
  OverlayCDB CDB(Base.get(), {}, std::string(""));
  EXPECT_THAT(CDB.getCompileCommand(testPath("foo.cc"))->CommandLine,
              AllOf(Contains(testPath("foo.cc")), Contains("-DA=1")));
  EXPECT_EQ(CDB.getCompileCommand(testPath("missing.cc")), llvm::None);

  auto Override = cmd(testPath("foo.cc"), "-DA=3");
  CDB.setCompileCommand(testPath("foo.cc"), Override);
  EXPECT_THAT(CDB.getCompileCommand(testPath("foo.cc"))->CommandLine,
              Contains("-DA=3"));
  EXPECT_EQ(CDB.getCompileCommand(testPath("missing.cc")), llvm::None);
  CDB.setCompileCommand(testPath("missing.cc"), Override);
  EXPECT_THAT(CDB.getCompileCommand(testPath("missing.cc"))->CommandLine,
              Contains("-DA=3"));
}

TEST_F(OverlayCDBTest, GetFallbackCommand) {
  OverlayCDB CDB(Base.get(), {"-DA=4"});
  EXPECT_THAT(CDB.getFallbackCommand(testPath("bar.cc")).CommandLine,
              ElementsAre("clang", "-DA=2", testPath("bar.cc"), "-DA=4",
                          "-fsyntax-only", StartsWith("-resource-dir")));
}

TEST_F(OverlayCDBTest, NoBase) {
  OverlayCDB CDB(nullptr, {"-DA=6"}, std::string(""));
  EXPECT_EQ(CDB.getCompileCommand(testPath("bar.cc")), None);
  auto Override = cmd(testPath("bar.cc"), "-DA=5");
  CDB.setCompileCommand(testPath("bar.cc"), Override);
  EXPECT_THAT(CDB.getCompileCommand(testPath("bar.cc"))->CommandLine,
              Contains("-DA=5"));

  EXPECT_THAT(CDB.getFallbackCommand(testPath("foo.cc")).CommandLine,
              ElementsAre(EndsWith("clang"), testPath("foo.cc"), "-DA=6",
                          "-fsyntax-only"));
}

TEST_F(OverlayCDBTest, Watch) {
  OverlayCDB Inner(nullptr);
  OverlayCDB Outer(&Inner);

  std::vector<std::vector<std::string>> Changes;
  auto Sub = Outer.watch([&](const std::vector<std::string> &ChangedFiles) {
    Changes.push_back(ChangedFiles);
  });

  Inner.setCompileCommand("A.cpp", tooling::CompileCommand());
  Outer.setCompileCommand("B.cpp", tooling::CompileCommand());
  Inner.setCompileCommand("A.cpp", llvm::None);
  Outer.setCompileCommand("C.cpp", llvm::None);
  EXPECT_THAT(Changes, ElementsAre(ElementsAre("A.cpp"), ElementsAre("B.cpp"),
                                   ElementsAre("A.cpp"), ElementsAre("C.cpp")));
}

TEST_F(OverlayCDBTest, Adjustments) {
  OverlayCDB CDB(Base.get(), {}, std::string(""));
  auto Cmd = CDB.getCompileCommand(testPath("foo.cc")).getValue();
  // Delete the file name.
  Cmd.CommandLine.pop_back();

  // Check dependency file commands are dropped.
  Cmd.CommandLine.push_back("-MF");
  Cmd.CommandLine.push_back("random-dependency");

  // Check plugin-related commands are dropped.
  Cmd.CommandLine.push_back("-Xclang");
  Cmd.CommandLine.push_back("-load");
  Cmd.CommandLine.push_back("-Xclang");
  Cmd.CommandLine.push_back("random-plugin");

  Cmd.CommandLine.push_back("-DA=5");
  Cmd.CommandLine.push_back(Cmd.Filename);

  CDB.setCompileCommand(testPath("foo.cc"), Cmd);

  EXPECT_THAT(CDB.getCompileCommand(testPath("foo.cc"))->CommandLine,
              AllOf(Contains("-fsyntax-only"), Contains("-DA=5"),
                    Contains(testPath("foo.cc")), Not(Contains("-MF")),
                    Not(Contains("random-dependency")),
                    Not(Contains("-Xclang")), Not(Contains("-load")),
                    Not(Contains("random-plugin"))));
}

} // namespace
} // namespace clangd
} // namespace clang
