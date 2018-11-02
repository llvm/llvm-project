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

static tooling::CompileCommand cmd(StringRef File, StringRef Arg) {
  return tooling::CompileCommand(testRoot(), File, {"clang", Arg, File}, "");
}

class OverlayCDBTest : public ::testing::Test {
  class BaseCDB : public GlobalCompilationDatabase {
  public:
    Optional<tooling::CompileCommand>
    getCompileCommand(StringRef File) const override {
      if (File == testPath("foo.cc"))
        return cmd(File, "-DA=1");
      return None;
    }

    tooling::CompileCommand getFallbackCommand(StringRef File) const override {
      return cmd(File, "-DA=2");
    }
  };

protected:
  OverlayCDBTest() : Base(llvm::make_unique<BaseCDB>()) {}
  std::unique_ptr<GlobalCompilationDatabase> Base;
};

TEST_F(OverlayCDBTest, GetCompileCommand) {
  OverlayCDB CDB(Base.get());
  EXPECT_EQ(CDB.getCompileCommand(testPath("foo.cc")),
            Base->getCompileCommand(testPath("foo.cc")));
  EXPECT_EQ(CDB.getCompileCommand(testPath("missing.cc")), llvm::None);

  auto Override = cmd(testPath("foo.cc"), "-DA=3");
  CDB.setCompileCommand(testPath("foo.cc"), Override);
  EXPECT_EQ(CDB.getCompileCommand(testPath("foo.cc")), Override);
  EXPECT_EQ(CDB.getCompileCommand(testPath("missing.cc")), llvm::None);
  CDB.setCompileCommand(testPath("missing.cc"), Override);
  EXPECT_EQ(CDB.getCompileCommand(testPath("missing.cc")), Override);
}

TEST_F(OverlayCDBTest, GetFallbackCommand) {
  OverlayCDB CDB(Base.get(), {"-DA=4"});
  EXPECT_THAT(CDB.getFallbackCommand(testPath("bar.cc")).CommandLine,
              ElementsAre("clang", "-DA=2", testPath("bar.cc"), "-DA=4"));
}

TEST_F(OverlayCDBTest, NoBase) {
  OverlayCDB CDB(nullptr, {"-DA=6"});
  EXPECT_EQ(CDB.getCompileCommand(testPath("bar.cc")), None);
  auto Override = cmd(testPath("bar.cc"), "-DA=5");
  CDB.setCompileCommand(testPath("bar.cc"), Override);
  EXPECT_EQ(CDB.getCompileCommand(testPath("bar.cc")), Override);

  EXPECT_THAT(CDB.getFallbackCommand(testPath("foo.cc")).CommandLine,
              ElementsAre("clang", testPath("foo.cc"), "-DA=6"));
}

} // namespace
} // namespace clangd
} // namespace clang
