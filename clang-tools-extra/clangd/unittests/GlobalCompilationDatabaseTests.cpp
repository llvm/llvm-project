//===-- GlobalCompilationDatabaseTests.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"

#include "Config.h"
#include "Matchers.h"
#include "TestFS.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
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
#include <chrono>
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
  MockFS TFS;
  DirectoryBasedGlobalCompilationDatabase DB(TFS);
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
  return tooling::CompileCommand(
      testRoot(), File, {"clang", std::string(Arg), std::string(File)}, "");
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
  MockFS FS;
  FS.Files[testPath("compile_commands.json")] =
      llvm::formatv(CDBOuter, llvm::sys::path::convert_to_slash(testRoot()));
  FS.Files[testPath("build/compile_commands.json")] =
      llvm::formatv(CDBInner, llvm::sys::path::convert_to_slash(testRoot()));
  FS.Files[testPath("foo/compile_flags.txt")] = "-DFOO";

  // Note that gen2.cc goes missing with our following model, not sure this
  // happens in practice though.
  {
    SCOPED_TRACE("Default ancestor scanning");
    DirectoryBasedGlobalCompilationDatabase DB(FS);
    std::vector<std::string> DiscoveredFiles;
    auto Sub =
        DB.watch([&DiscoveredFiles](const std::vector<std::string> Changes) {
          DiscoveredFiles = Changes;
        });

    DB.getCompileCommand(testPath("build/../a.cc"));
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles, UnorderedElementsAre(AllOf(
                                     EndsWith("a.cc"), Not(HasSubstr("..")))));
    DiscoveredFiles.clear();

    DB.getCompileCommand(testPath("build/gen.cc"));
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles, UnorderedElementsAre(EndsWith("gen.cc")));
  }

  {
    SCOPED_TRACE("With config");
    DirectoryBasedGlobalCompilationDatabase::Options Opts(FS);
    Opts.ContextProvider = [&](llvm::StringRef Path) {
      Config Cfg;
      if (Path.endswith("a.cc")) {
        // a.cc uses another directory's CDB, so it won't be discovered.
        Cfg.CompileFlags.CDBSearch.Policy = Config::CDBSearchSpec::FixedDir;
        Cfg.CompileFlags.CDBSearch.FixedCDBPath = testPath("foo");
      } else if (Path.endswith("gen.cc")) {
        // gen.cc has CDB search disabled, so it won't be discovered.
        Cfg.CompileFlags.CDBSearch.Policy = Config::CDBSearchSpec::NoCDBSearch;
      } else if (Path.endswith("gen2.cc")) {
        // gen2.cc explicitly lists this directory, so it will be discovered.
        Cfg.CompileFlags.CDBSearch.Policy = Config::CDBSearchSpec::FixedDir;
        Cfg.CompileFlags.CDBSearch.FixedCDBPath = testRoot();
      }
      return Context::current().derive(Config::Key, std::move(Cfg));
    };
    DirectoryBasedGlobalCompilationDatabase DB(Opts);
    std::vector<std::string> DiscoveredFiles;
    auto Sub =
        DB.watch([&DiscoveredFiles](const std::vector<std::string> Changes) {
          DiscoveredFiles = Changes;
        });

    // Does not use the root CDB, so no broadcast.
    auto Cmd = DB.getCompileCommand(testPath("build/../a.cc"));
    ASSERT_TRUE(Cmd.hasValue());
    EXPECT_THAT(Cmd->CommandLine, Contains("-DFOO")) << "a.cc uses foo/ CDB";
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles, IsEmpty()) << "Root CDB not discovered yet";

    // No special config for b.cc, so we trigger broadcast of the root CDB.
    DB.getCompileCommand(testPath("b.cc"));
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles, ElementsAre(testPath("build/gen2.cc")));
    DiscoveredFiles.clear();

    // No CDB search so no discovery/broadcast triggered for build/ CDB.
    DB.getCompileCommand(testPath("build/gen.cc"));
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles, IsEmpty());
  }

  {
    SCOPED_TRACE("With custom compile commands dir");
    DirectoryBasedGlobalCompilationDatabase::Options Opts(FS);
    Opts.CompileCommandsDir = testRoot();
    DirectoryBasedGlobalCompilationDatabase DB(Opts);
    std::vector<std::string> DiscoveredFiles;
    auto Sub =
        DB.watch([&DiscoveredFiles](const std::vector<std::string> Changes) {
          DiscoveredFiles = Changes;
        });

    DB.getCompileCommand(testPath("a.cc"));
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles,
                UnorderedElementsAre(EndsWith("a.cc"), EndsWith("gen.cc"),
                                     EndsWith("gen2.cc")));
    DiscoveredFiles.clear();

    DB.getCompileCommand(testPath("build/gen.cc"));
    ASSERT_TRUE(DB.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_THAT(DiscoveredFiles, IsEmpty());
  }
}

TEST(GlobalCompilationDatabaseTest, BuildDir) {
  MockFS FS;
  auto Command = [&](llvm::StringRef Relative) {
    DirectoryBasedGlobalCompilationDatabase::Options Opts(FS);
    return DirectoryBasedGlobalCompilationDatabase(Opts)
        .getCompileCommand(testPath(Relative))
        .getValueOr(tooling::CompileCommand())
        .CommandLine;
  };
  EXPECT_THAT(Command("x/foo.cc"), IsEmpty());
  const char *const CDB =
      R"cdb(
      [
        {
          "file": "{0}/x/foo.cc",
          "command": "clang -DXYZZY {0}/x/foo.cc",
          "directory": "{0}",
        },
        {
          "file": "{0}/bar.cc",
          "command": "clang -DXYZZY {0}/bar.cc",
          "directory": "{0}",
        }
      ]
      )cdb";
  FS.Files[testPath("x/build/compile_commands.json")] =
      llvm::formatv(CDB, llvm::sys::path::convert_to_slash(testRoot()));
  EXPECT_THAT(Command("x/foo.cc"), Contains("-DXYZZY"));
  EXPECT_THAT(Command("bar.cc"), IsEmpty())
      << "x/build/compile_flags.json only applicable to x/";
}

TEST(GlobalCompilationDatabaseTest, CompileFlagsDirectory) {
  MockFS FS;
  FS.Files[testPath("x/compile_flags.txt")] = "-DFOO";
  DirectoryBasedGlobalCompilationDatabase CDB(FS);
  auto Commands = CDB.getCompileCommand(testPath("x/y.cpp"));
  ASSERT_TRUE(Commands.hasValue());
  EXPECT_THAT(Commands.getValue().CommandLine, Contains("-DFOO"));
  // Make sure we pick the right working directory.
  EXPECT_EQ(testPath("x"), Commands.getValue().Directory);
}

MATCHER_P(hasArg, Flag, "") {
  if (!arg.hasValue()) {
    *result_listener << "command is null";
    return false;
  }
  if (!llvm::is_contained(arg->CommandLine, Flag)) {
    *result_listener << "flags are " << printArgv(arg->CommandLine);
    return false;
  }
  return true;
}

TEST(GlobalCompilationDatabaseTest, Config) {
  MockFS FS;
  FS.Files[testPath("x/compile_flags.txt")] = "-DX";
  FS.Files[testPath("x/y/z/compile_flags.txt")] = "-DZ";

  Config::CDBSearchSpec Spec;
  DirectoryBasedGlobalCompilationDatabase::Options Opts(FS);
  Opts.ContextProvider = [&](llvm::StringRef Path) {
    Config C;
    C.CompileFlags.CDBSearch = Spec;
    return Context::current().derive(Config::Key, std::move(C));
  };
  DirectoryBasedGlobalCompilationDatabase CDB(Opts);

  // Default ancestor behavior.
  EXPECT_FALSE(CDB.getCompileCommand(testPath("foo.cc")));
  EXPECT_THAT(CDB.getCompileCommand(testPath("x/foo.cc")), hasArg("-DX"));
  EXPECT_THAT(CDB.getCompileCommand(testPath("x/y/foo.cc")), hasArg("-DX"));
  EXPECT_THAT(CDB.getCompileCommand(testPath("x/y/z/foo.cc")), hasArg("-DZ"));

  Spec.Policy = Config::CDBSearchSpec::NoCDBSearch;
  EXPECT_FALSE(CDB.getCompileCommand(testPath("foo.cc")));
  EXPECT_FALSE(CDB.getCompileCommand(testPath("x/foo.cc")));
  EXPECT_FALSE(CDB.getCompileCommand(testPath("x/y/foo.cc")));
  EXPECT_FALSE(CDB.getCompileCommand(testPath("x/y/z/foo.cc")));

  Spec.Policy = Config::CDBSearchSpec::FixedDir;
  Spec.FixedCDBPath = testPath("w"); // doesn't exist
  EXPECT_FALSE(CDB.getCompileCommand(testPath("foo.cc")));
  EXPECT_FALSE(CDB.getCompileCommand(testPath("x/foo.cc")));
  EXPECT_FALSE(CDB.getCompileCommand(testPath("x/y/foo.cc")));
  EXPECT_FALSE(CDB.getCompileCommand(testPath("x/y/z/foo.cc")));

  Spec.FixedCDBPath = testPath("x/y/z");
  EXPECT_THAT(CDB.getCompileCommand(testPath("foo.cc")), hasArg("-DZ"));
  EXPECT_THAT(CDB.getCompileCommand(testPath("x/foo.cc")), hasArg("-DZ"));
  EXPECT_THAT(CDB.getCompileCommand(testPath("x/y/foo.cc")), hasArg("-DZ"));
  EXPECT_THAT(CDB.getCompileCommand(testPath("x/y/z/foo.cc")), hasArg("-DZ"));
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
  EXPECT_FALSE(DB.getProjectInfo(File));
}

TEST_F(OverlayCDBTest, GetProjectInfo) {
  OverlayCDB DB(Base.get());
  Path File = testPath("foo.cc");
  Path Header = testPath("foo.h");

  EXPECT_EQ(DB.getProjectInfo(File)->SourceRoot, testRoot());
  EXPECT_EQ(DB.getProjectInfo(Header)->SourceRoot, testRoot());

  // Shouldn't change after an override.
  DB.setCompileCommand(File, tooling::CompileCommand());
  EXPECT_EQ(DB.getProjectInfo(File)->SourceRoot, testRoot());
  EXPECT_EQ(DB.getProjectInfo(Header)->SourceRoot, testRoot());
}
} // namespace

// Friend test has access to internals.
class DirectoryBasedGlobalCompilationDatabaseCacheTest
    : public ::testing::Test {
protected:
  std::shared_ptr<const tooling::CompilationDatabase>
  lookupCDB(const DirectoryBasedGlobalCompilationDatabase &GDB,
            llvm::StringRef Path,
            std::chrono::steady_clock::time_point FreshTime) {
    DirectoryBasedGlobalCompilationDatabase::CDBLookupRequest Req;
    Req.FileName = Path;
    Req.FreshTime = Req.FreshTimeMissing = FreshTime;
    if (auto Result = GDB.lookupCDB(Req))
      return std::move(Result->CDB);
    return nullptr;
  }
};

// Matches non-null CDBs which include the specified flag.
MATCHER_P2(hasFlag, Flag, Path, "") {
  if (arg == nullptr)
    return false;
  auto Cmds = arg->getCompileCommands(Path);
  if (Cmds.empty()) {
    *result_listener << "yields no commands";
    return false;
  }
  if (!llvm::is_contained(Cmds.front().CommandLine, Flag)) {
    *result_listener << "flags are: " << printArgv(Cmds.front().CommandLine);
    return false;
  }
  return true;
}

auto hasFlag(llvm::StringRef Flag) { return hasFlag(Flag, "dummy.cc"); }

TEST_F(DirectoryBasedGlobalCompilationDatabaseCacheTest, Cacheable) {
  MockFS FS;
  auto Stale = std::chrono::steady_clock::now() - std::chrono::minutes(1);
  auto Fresh = std::chrono::steady_clock::now() + std::chrono::hours(24);

  DirectoryBasedGlobalCompilationDatabase GDB(FS);
  FS.Files["compile_flags.txt"] = "-DROOT";
  auto Root = lookupCDB(GDB, testPath("foo/test.cc"), Stale);
  EXPECT_THAT(Root, hasFlag("-DROOT"));

  // Add a compilation database to a subdirectory - CDB loaded.
  FS.Files["foo/compile_flags.txt"] = "-DFOO";
  EXPECT_EQ(Root, lookupCDB(GDB, testPath("foo/test.cc"), Stale))
      << "cache still valid";
  auto Foo = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_THAT(Foo, hasFlag("-DFOO")) << "new cdb loaded";
  EXPECT_EQ(Foo, lookupCDB(GDB, testPath("foo/test.cc"), Stale))
      << "new cdb in cache";

  // Mtime changed, but no content change - CDB not reloaded.
  ++FS.Timestamps["foo/compile_flags.txt"];
  auto FooAgain = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_EQ(Foo, FooAgain) << "Same content, read but not reloaded";
  // Content changed, but not size or mtime - CDB not reloaded.
  FS.Files["foo/compile_flags.txt"] = "-DBAR";
  auto FooAgain2 = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_EQ(Foo, FooAgain2) << "Same filesize, change not detected";
  // Mtime change forces a re-read, and we notice the different content.
  ++FS.Timestamps["foo/compile_flags.txt"];
  auto Bar = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_THAT(Bar, hasFlag("-DBAR")) << "refreshed with mtime change";

  // Size and content both change - CDB reloaded.
  FS.Files["foo/compile_flags.txt"] = "-DFOOBAR";
  EXPECT_EQ(Bar, lookupCDB(GDB, testPath("foo/test.cc"), Stale))
      << "cache still valid";
  auto FooBar = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_THAT(FooBar, hasFlag("-DFOOBAR")) << "cdb reloaded";

  // compile_commands.json takes precedence over compile_flags.txt.
  FS.Files["foo/compile_commands.json"] =
      llvm::formatv(R"json([{
    "file": "{0}/foo/dummy.cc",
    "command": "clang -DBAZ dummy.cc",
    "directory": "{0}/foo",
  }])json",
                    llvm::sys::path::convert_to_slash(testRoot()));
  EXPECT_EQ(FooBar, lookupCDB(GDB, testPath("foo/test.cc"), Stale))
      << "cache still valid";
  auto Baz = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_THAT(Baz, hasFlag("-DBAZ", testPath("foo/dummy.cc")))
      << "compile_commands overrides compile_flags";

  // Removing compile_commands.json reveals compile_flags.txt again.
  // However this *does* cause a CDB reload (we cache only one CDB per dir).
  FS.Files.erase("foo/compile_commands.json");
  auto FoobarAgain = lookupCDB(GDB, testPath("foo/test.cc"), Fresh);
  EXPECT_THAT(FoobarAgain, hasFlag("-DFOOBAR")) << "reloaded compile_flags";
  EXPECT_NE(FoobarAgain, FooBar) << "CDB discarded (shadowed within directory)";

  // Removing the directory's CDB leaves the parent CDB active.
  // The parent CDB is *not* reloaded (we cache the CDB per-directory).
  FS.Files.erase("foo/compile_flags.txt");
  EXPECT_EQ(Root, lookupCDB(GDB, testPath("foo/test.cc"), Fresh))
      << "CDB retained (shadowed by another directory)";
}

} // namespace clangd
} // namespace clang
