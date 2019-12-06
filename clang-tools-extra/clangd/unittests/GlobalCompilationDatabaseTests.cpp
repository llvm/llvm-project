//===-- GlobalCompilationDatabaseTests.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"

#include "Path.h"
#include "TestFS.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <fstream>
#include <string>

namespace clang {
namespace clangd {
namespace {
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::StartsWith;
using ::testing::UnorderedElementsAre;

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
  Cmd = DB.getFallbackCommand(testPath("foo/bar"));
  EXPECT_THAT(Cmd.CommandLine, ElementsAre("clang", "-xobjective-c++-header",
                                           testPath("foo/bar")));
}

static tooling::CompileCommand cmd(llvm::StringRef File, llvm::StringRef Arg) {
  return tooling::CompileCommand(testRoot(), File, {"clang", Arg, File}, "");
}

class OverlayCDBTest : public ::testing::Test {
  class BaseCDB : public GlobalCompilationDatabase {
  public:
    llvm::Optional<tooling::CompileCommand>
    getCompileCommand(llvm::StringRef File) const override {
      if (File == testPath("foo.cc"))
        return cmd(File, "-DA=1");
      return None;
    }

    tooling::CompileCommand
    getFallbackCommand(llvm::StringRef File) const override {
      return cmd(File, "-DA=2");
    }

    llvm::Optional<ProjectInfo> getProjectInfo(PathRef File) const override {
      return ProjectInfo{testRoot()};
    }
  };

protected:
  OverlayCDBTest() : Base(std::make_unique<BaseCDB>()) {}
  std::unique_ptr<GlobalCompilationDatabase> Base;
};

TEST_F(OverlayCDBTest, GetCompileCommand) {
  OverlayCDB CDB(Base.get());
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
              ElementsAre("clang", "-DA=2", testPath("bar.cc"), "-DA=4"));
}

TEST_F(OverlayCDBTest, NoBase) {
  OverlayCDB CDB(nullptr, {"-DA=6"});
  EXPECT_EQ(CDB.getCompileCommand(testPath("bar.cc")), None);
  auto Override = cmd(testPath("bar.cc"), "-DA=5");
  CDB.setCompileCommand(testPath("bar.cc"), Override);
  EXPECT_THAT(CDB.getCompileCommand(testPath("bar.cc"))->CommandLine,
              Contains("-DA=5"));

  EXPECT_THAT(CDB.getFallbackCommand(testPath("foo.cc")).CommandLine,
              ElementsAre("clang", testPath("foo.cc"), "-DA=6"));
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
  OverlayCDB CDB(Base.get(), {"-DFallback"},
                 [](const std::vector<std::string> &Cmd, llvm::StringRef File) {
                   auto Ret = Cmd;
                   Ret.push_back(
                       ("-DAdjust_" + llvm::sys::path::filename(File)).str());
                   return Ret;
                 });
  // Command from underlying gets adjusted.
  auto Cmd = CDB.getCompileCommand(testPath("foo.cc")).getValue();
  EXPECT_THAT(Cmd.CommandLine, ElementsAre("clang", "-DA=1", testPath("foo.cc"),
                                           "-DAdjust_foo.cc"));

  // Command from overlay gets adjusted.
  tooling::CompileCommand BarCommand;
  BarCommand.Filename = testPath("bar.cc");
  BarCommand.CommandLine = {"clang++", "-DB=1", testPath("bar.cc")};
  CDB.setCompileCommand(testPath("bar.cc"), BarCommand);
  Cmd = CDB.getCompileCommand(testPath("bar.cc")).getValue();
  EXPECT_THAT(
      Cmd.CommandLine,
      ElementsAre("clang++", "-DB=1", testPath("bar.cc"), "-DAdjust_bar.cc"));

  // Fallback gets adjusted.
  Cmd = CDB.getFallbackCommand("baz.cc");
  EXPECT_THAT(Cmd.CommandLine, ElementsAre("clang", "-DA=2", "baz.cc",
                                           "-DFallback", "-DAdjust_baz.cc"));
}

TEST(GlobalCompilationDatabaseTest, DiscoveryWithNestedCDBs) {
  const char *const CDBOuter =
      R"cdb(
      [
        {
          "file": "a.cc",
          "command": "",
          "directory": "{0}",
        },
        {
          "file": "build/gen.cc",
          "command": "",
          "directory": "{0}",
        },
        {
          "file": "build/gen2.cc",
          "command": "",
          "directory": "{0}",
        }
      ]
      )cdb";
  const char *const CDBInner =
      R"cdb(
      [
        {
          "file": "gen.cc",
          "command": "",
          "directory": "{0}/build",
        }
      ]
      )cdb";
  class CleaningFS {
  public:
    llvm::SmallString<128> Root;

    CleaningFS() {
      EXPECT_FALSE(
          llvm::sys::fs::createUniqueDirectory("clangd-cdb-test", Root))
          << "Failed to create unique directory";
    }

    ~CleaningFS() {
      EXPECT_FALSE(llvm::sys::fs::remove_directories(Root))
          << "Failed to cleanup " << Root;
    }

    void registerFile(PathRef RelativePath, llvm::StringRef Contents) {
      llvm::SmallString<128> AbsPath(Root);
      llvm::sys::path::append(AbsPath, RelativePath);

      EXPECT_FALSE(llvm::sys::fs::create_directories(
          llvm::sys::path::parent_path(AbsPath)))
          << "Failed to create directories for: " << AbsPath;

      std::error_code EC;
      llvm::raw_fd_ostream OS(AbsPath, EC);
      EXPECT_FALSE(EC) << "Failed to open " << AbsPath << " for writing";
      OS << llvm::formatv(Contents.data(),
                          llvm::sys::path::convert_to_slash(Root));
      OS.close();

      EXPECT_FALSE(OS.has_error());
    }
  };

  CleaningFS FS;
  FS.registerFile("compile_commands.json", CDBOuter);
  FS.registerFile("build/compile_commands.json", CDBInner);
  llvm::SmallString<128> File;

  // Note that gen2.cc goes missing with our following model, not sure this
  // happens in practice though.
  {
    DirectoryBasedGlobalCompilationDatabase DB(llvm::None);
    std::vector<std::string> DiscoveredFiles;
    auto Sub =
        DB.watch([&DiscoveredFiles](const std::vector<std::string> Changes) {
          DiscoveredFiles = Changes;
        });

    File = FS.Root;
    llvm::sys::path::append(File, "build", "..", "a.cc");
    DB.getCompileCommand(File.str());
    EXPECT_THAT(DiscoveredFiles, UnorderedElementsAre(AllOf(
                                     EndsWith("a.cc"), Not(HasSubstr("..")))));
    DiscoveredFiles.clear();

    File = FS.Root;
    llvm::sys::path::append(File, "build", "gen.cc");
    DB.getCompileCommand(File.str());
    EXPECT_THAT(DiscoveredFiles, UnorderedElementsAre(EndsWith("gen.cc")));
  }

  // With a custom compile commands dir.
  {
    DirectoryBasedGlobalCompilationDatabase DB(FS.Root.str().str());
    std::vector<std::string> DiscoveredFiles;
    auto Sub =
        DB.watch([&DiscoveredFiles](const std::vector<std::string> Changes) {
          DiscoveredFiles = Changes;
        });

    File = FS.Root;
    llvm::sys::path::append(File, "a.cc");
    DB.getCompileCommand(File.str());
    EXPECT_THAT(DiscoveredFiles,
                UnorderedElementsAre(EndsWith("a.cc"), EndsWith("gen.cc"),
                                     EndsWith("gen2.cc")));
    DiscoveredFiles.clear();

    File = FS.Root;
    llvm::sys::path::append(File, "build", "gen.cc");
    DB.getCompileCommand(File.str());
    EXPECT_THAT(DiscoveredFiles, IsEmpty());
  }
}

TEST(GlobalCompilationDatabaseTest, NonCanonicalFilenames) {
  OverlayCDB DB(nullptr);
  std::vector<std::string> DiscoveredFiles;
  auto Sub =
      DB.watch([&DiscoveredFiles](const std::vector<std::string> Changes) {
        DiscoveredFiles = Changes;
      });

  llvm::SmallString<128> Root(testRoot());
  llvm::sys::path::append(Root, "build", "..", "a.cc");
  DB.setCompileCommand(Root.str(), tooling::CompileCommand());
  EXPECT_THAT(DiscoveredFiles, UnorderedElementsAre(testPath("a.cc")));
  DiscoveredFiles.clear();

  llvm::SmallString<128> File(testRoot());
  llvm::sys::path::append(File, "blabla", "..", "a.cc");

  EXPECT_TRUE(DB.getCompileCommand(File));
  EXPECT_TRUE(DB.getProjectInfo(File));
}

} // namespace
} // namespace clangd
} // namespace clang
