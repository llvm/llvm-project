//===-- CompileCommandsTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "Config.h"
#include "TestFS.h"
#include "support/Context.h"

#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::_;
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
  llvm::InitializeAllTargetInfos(); // As in ClangdMain
  std::string Target = getAnyTargetForTesting();
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");
  Mangler.ResourceDir = testPath("fake/resources");
  Mangler.Sysroot = testPath("fake/sysroot");
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {Target + "-clang++", "--", "foo.cc", "bar.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_THAT(Cmd.CommandLine,
              ElementsAre(testPath("fake/" + Target + "-clang++"),
                          "--target=" + Target, "--driver-mode=g++",
                          "-resource-dir=" + testPath("fake/resources"),
                          "-isysroot", testPath("fake/sysroot"), "--",
                          "foo.cc"));
}

TEST(CommandMangler, FilenameMismatch) {
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("clang");
  // Our compile flags refer to foo.cc...
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {"clang", "foo.cc"};
  // but we're applying it to foo.h...
  Mangler(Cmd, "foo.h");
  // so transferCompileCommand should add -x c++-header to preserve semantics.
  EXPECT_THAT(Cmd.CommandLine, ElementsAre(testPath("clang"), "-x",
                                           "c++-header", "--", "foo.h"));
}

TEST(CommandMangler, ResourceDir) {
  auto Mangler = CommandMangler::forTests();
  Mangler.ResourceDir = testPath("fake/resources");
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {"clang++", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_THAT(Cmd.CommandLine,
              Contains("-resource-dir=" + testPath("fake/resources")));
}

TEST(CommandMangler, Sysroot) {
  auto Mangler = CommandMangler::forTests();
  Mangler.Sysroot = testPath("fake/sysroot");

  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {"clang++", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_THAT(llvm::join(Cmd.CommandLine, " "),
              HasSubstr("-isysroot " + testPath("fake/sysroot")));
}

TEST(CommandMangler, ClangPath) {
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");

  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {"clang++", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_EQ(testPath("fake/clang++"), Cmd.CommandLine.front());

  Cmd.CommandLine = {"unknown-binary", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_EQ(testPath("fake/unknown-binary"), Cmd.CommandLine.front());

  Cmd.CommandLine = {testPath("path/clang++"), "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_EQ(testPath("path/clang++"), Cmd.CommandLine.front());

  Cmd.CommandLine = {"foo/unknown-binary", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_EQ("foo/unknown-binary", Cmd.CommandLine.front());
}

// Only run the PATH/symlink resolving test on unix, we need to fiddle
// with permissions and environment variables...
#ifdef LLVM_ON_UNIX
MATCHER(ok, "") {
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
              ok());
  // /var/tmp is a symlink on Mac. Resolve it so we're asserting the right path.
  ASSERT_THAT(llvm::sys::fs::real_path(TempDir.str(), TempDir), ok());
  auto CleanDir = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(TempDir); });
  ASSERT_THAT(llvm::sys::fs::create_directory(TempDir + "/bin"), ok());
  ASSERT_THAT(llvm::sys::fs::create_directory(TempDir + "/lib"), ok());
  int FD;
  ASSERT_THAT(llvm::sys::fs::openFileForWrite(TempDir + "/lib/bar", FD), ok());
  ASSERT_THAT(llvm::sys::Process::SafelyCloseFileDescriptor(FD), ok());
  ::chmod((TempDir + "/lib/bar").str().c_str(), 0755); // executable
  ASSERT_THAT(
      llvm::sys::fs::create_link(TempDir + "/lib/bar", TempDir + "/bin/foo"),
      ok());

  // Test the case where the driver is an absolute path to a symlink.
  auto Mangler = CommandMangler::forTests();
  Mangler.ClangPath = testPath("fake/clang");
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {(TempDir + "/bin/foo").str(), "foo.cc"};
  Mangler(Cmd, "foo.cc");
  // Directory based on resolved symlink, basename preserved.
  EXPECT_EQ((TempDir + "/lib/foo").str(), Cmd.CommandLine.front());

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
  Cmd.CommandLine = {"foo", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  // Found the symlink and resolved the path as above.
  EXPECT_EQ((TempDir + "/lib/foo").str(), Cmd.CommandLine.front());

  // Symlink not resolved with -no-canonical-prefixes.
  Cmd.CommandLine = {"foo", "-no-canonical-prefixes", "foo.cc"};
  Mangler(Cmd, "foo.cc");
  EXPECT_EQ((TempDir + "/bin/foo").str(), Cmd.CommandLine.front());
}
#endif

TEST(CommandMangler, ConfigEdits) {
  auto Mangler = CommandMangler::forTests();
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {"clang++", "foo.cc"};
  {
    Config Cfg;
    Cfg.CompileFlags.Edits.push_back([](std::vector<std::string> &Argv) {
      for (auto &Arg : Argv)
        for (char &C : Arg)
          C = llvm::toUpper(C);
    });
    Cfg.CompileFlags.Edits.push_back([](std::vector<std::string> &Argv) {
      Argv = tooling::getInsertArgumentAdjuster("--hello")(Argv, "");
    });
    WithContextValue WithConfig(Config::Key, std::move(Cfg));
    Mangler(Cmd, "foo.cc");
  }
  // Edits are applied in given order and before other mangling and they always
  // go before filename. `--driver-mode=g++` here is in lower case because
  // options inserted by addTargetAndModeForProgramName are not editable,
  // see discussion in https://reviews.llvm.org/D138546
  EXPECT_THAT(Cmd.CommandLine,
              ElementsAre(_, "--driver-mode=g++", "--hello", "--", "FOO.CC"));
}

static std::string strip(llvm::StringRef Arg, llvm::StringRef Argv) {
  llvm::SmallVector<llvm::StringRef> Parts;
  llvm::SplitString(Argv, Parts);
  std::vector<std::string> Args = {Parts.begin(), Parts.end()};
  ArgStripper S;
  S.strip(Arg);
  S.process(Args);
  return printArgv(Args);
}

TEST(ArgStripperTest, Spellings) {
  // May use alternate prefixes.
  EXPECT_EQ(strip("-pedantic", "clang -pedantic foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-pedantic", "clang --pedantic foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("--pedantic", "clang -pedantic foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("--pedantic", "clang --pedantic foo.cc"), "clang foo.cc");
  // May use alternate names.
  EXPECT_EQ(strip("-x", "clang -x c++ foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-x", "clang --language=c++ foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("--language=", "clang -x c++ foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("--language=", "clang --language=c++ foo.cc"),
            "clang foo.cc");
}

TEST(ArgStripperTest, UnknownFlag) {
  EXPECT_EQ(strip("-xyzzy", "clang -xyzzy foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-xyz*", "clang -xyzzy foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-xyzzy", "clang -Xclang -xyzzy foo.cc"), "clang foo.cc");
}

TEST(ArgStripperTest, Xclang) {
  // Flags may be -Xclang escaped.
  EXPECT_EQ(strip("-ast-dump", "clang -Xclang -ast-dump foo.cc"),
            "clang foo.cc");
  // Args may be -Xclang escaped.
  EXPECT_EQ(strip("-add-plugin", "clang -Xclang -add-plugin -Xclang z foo.cc"),
            "clang foo.cc");
}

TEST(ArgStripperTest, ClangCL) {
  // /I is a synonym for -I in clang-cl mode only.
  // Not stripped by default.
  EXPECT_EQ(strip("-I", "clang -I /usr/inc /Interesting/file.cc"),
            "clang /Interesting/file.cc");
  // Stripped when invoked as clang-cl.
  EXPECT_EQ(strip("-I", "clang-cl -I /usr/inc /Interesting/file.cc"),
            "clang-cl");
  // Stripped when invoked as CL.EXE
  EXPECT_EQ(strip("-I", "CL.EXE -I /usr/inc /Interesting/file.cc"), "CL.EXE");
  // Stripped when passed --driver-mode=cl.
  EXPECT_EQ(strip("-I", "cc -I /usr/inc /Interesting/file.cc --driver-mode=cl"),
            "cc --driver-mode=cl");
}

TEST(ArgStripperTest, ArgStyles) {
  // Flag
  EXPECT_EQ(strip("-Qn", "clang -Qn foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-Qn", "clang -QnZ foo.cc"), "clang -QnZ foo.cc");
  // Joined
  EXPECT_EQ(strip("-std=", "clang -std= foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-std=", "clang -std=c++11 foo.cc"), "clang foo.cc");
  // Separate
  EXPECT_EQ(strip("-mllvm", "clang -mllvm X foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-mllvm", "clang -mllvmX foo.cc"), "clang -mllvmX foo.cc");
  // RemainingArgsJoined
  EXPECT_EQ(strip("/link", "clang-cl /link b c d foo.cc"), "clang-cl");
  EXPECT_EQ(strip("/link", "clang-cl /linka b c d foo.cc"), "clang-cl");
  // CommaJoined
  EXPECT_EQ(strip("-Wl,", "clang -Wl,x,y foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-Wl,", "clang -Wl, foo.cc"), "clang foo.cc");
  // MultiArg
  EXPECT_EQ(strip("-segaddr", "clang -segaddr a b foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-segaddr", "clang -segaddra b foo.cc"),
            "clang -segaddra b foo.cc");
  // JoinedOrSeparate
  EXPECT_EQ(strip("-G", "clang -GX foo.cc"), "clang foo.cc");
  EXPECT_EQ(strip("-G", "clang -G X foo.cc"), "clang foo.cc");
  // JoinedAndSeparate
  EXPECT_EQ(strip("-plugin-arg-", "clang -cc1 -plugin-arg-X Y foo.cc"),
            "clang -cc1 foo.cc");
  EXPECT_EQ(strip("-plugin-arg-", "clang -cc1 -plugin-arg- Y foo.cc"),
            "clang -cc1 foo.cc");
}

TEST(ArgStripperTest, EndOfList) {
  // When we hit the end-of-args prematurely, we don't crash.
  // We consume the incomplete args if we've matched the target option.
  EXPECT_EQ(strip("-I", "clang -Xclang"), "clang -Xclang");
  EXPECT_EQ(strip("-I", "clang -Xclang -I"), "clang");
  EXPECT_EQ(strip("-I", "clang -I -Xclang"), "clang");
  EXPECT_EQ(strip("-I", "clang -I"), "clang");
}

TEST(ArgStripperTest, Multiple) {
  ArgStripper S;
  S.strip("-o");
  S.strip("-c");
  std::vector<std::string> Args = {"clang", "-o", "foo.o", "foo.cc", "-c"};
  S.process(Args);
  EXPECT_THAT(Args, ElementsAre("clang", "foo.cc"));
}

TEST(ArgStripperTest, Warning) {
  {
    // -W is a flag name
    ArgStripper S;
    S.strip("-W");
    std::vector<std::string> Args = {"clang", "-Wfoo", "-Wno-bar", "-Werror",
                                     "foo.cc"};
    S.process(Args);
    EXPECT_THAT(Args, ElementsAre("clang", "foo.cc"));
  }
  {
    // -Wfoo is not a flag name, matched literally.
    ArgStripper S;
    S.strip("-Wunused");
    std::vector<std::string> Args = {"clang", "-Wunused", "-Wno-unused",
                                     "foo.cc"};
    S.process(Args);
    EXPECT_THAT(Args, ElementsAre("clang", "-Wno-unused", "foo.cc"));
  }
}

TEST(ArgStripperTest, Define) {
  {
    // -D is a flag name
    ArgStripper S;
    S.strip("-D");
    std::vector<std::string> Args = {"clang", "-Dfoo", "-Dbar=baz", "foo.cc"};
    S.process(Args);
    EXPECT_THAT(Args, ElementsAre("clang", "foo.cc"));
  }
  {
    // -Dbar is not: matched literally
    ArgStripper S;
    S.strip("-Dbar");
    std::vector<std::string> Args = {"clang", "-Dfoo", "-Dbar=baz", "foo.cc"};
    S.process(Args);
    EXPECT_THAT(Args, ElementsAre("clang", "-Dfoo", "-Dbar=baz", "foo.cc"));
    S.strip("-Dfoo");
    S.process(Args);
    EXPECT_THAT(Args, ElementsAre("clang", "-Dbar=baz", "foo.cc"));
    S.strip("-Dbar=*");
    S.process(Args);
    EXPECT_THAT(Args, ElementsAre("clang", "foo.cc"));
  }
}

TEST(ArgStripperTest, OrderDependent) {
  ArgStripper S;
  // If -include is stripped first, we see -pch as its arg and foo.pch remains.
  // To get this case right, we must process -include-pch first.
  S.strip("-include");
  S.strip("-include-pch");
  std::vector<std::string> Args = {"clang", "-include-pch", "foo.pch",
                                   "foo.cc"};
  S.process(Args);
  EXPECT_THAT(Args, ElementsAre("clang", "foo.cc"));
}

TEST(PrintArgvTest, All) {
  std::vector<llvm::StringRef> Args = {
      "one", "two", "thr ee", "f\"o\"ur", "fi\\ve", "$"
  };
  const char *Expected = R"(one two "thr ee" "f\"o\"ur" "fi\\ve" $)";
  EXPECT_EQ(Expected, printArgv(Args));
}

TEST(CommandMangler, InputsAfterDashDash) {
  const auto Mangler = CommandMangler::forTests();
  {
    tooling::CompileCommand Cmd;
    Cmd.CommandLine = {"clang", "/Users/foo.cc"};
    Mangler(Cmd, "/Users/foo.cc");
    EXPECT_THAT(llvm::ArrayRef(Cmd.CommandLine).take_back(2),
                ElementsAre("--", "/Users/foo.cc"));
    EXPECT_THAT(llvm::ArrayRef(Cmd.CommandLine).drop_back(2),
                Not(Contains("/Users/foo.cc")));
  }
  // In CL mode /U triggers an undef operation, hence `/Users/foo.cc` shouldn't
  // be interpreted as a file.
  {
    tooling::CompileCommand Cmd;
    Cmd.CommandLine = {"clang", "--driver-mode=cl", "bar.cc", "/Users/foo.cc"};
    Mangler(Cmd, "bar.cc");
    EXPECT_THAT(llvm::ArrayRef(Cmd.CommandLine).take_back(2),
                ElementsAre("--", "bar.cc"));
    EXPECT_THAT(llvm::ArrayRef(Cmd.CommandLine).drop_back(2),
                Not(Contains("bar.cc")));
  }
  // All inputs but the main file is dropped.
  {
    tooling::CompileCommand Cmd;
    Cmd.CommandLine = {"clang", "foo.cc", "bar.cc"};
    Mangler(Cmd, "baz.cc");
    EXPECT_THAT(llvm::ArrayRef(Cmd.CommandLine).take_back(2),
                ElementsAre("--", "baz.cc"));
    EXPECT_THAT(
        llvm::ArrayRef(Cmd.CommandLine).drop_back(2),
        testing::AllOf(Not(Contains("foo.cc")), Not(Contains("bar.cc"))));
  }
}

TEST(CommandMangler, StripsMultipleArch) {
  const auto Mangler = CommandMangler::forTests();
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {"clang", "-arch", "foo", "-arch", "bar", "/Users/foo.cc"};
  Mangler(Cmd, "/Users/foo.cc");
  EXPECT_EQ(llvm::count_if(Cmd.CommandLine,
                           [](llvm::StringRef Arg) { return Arg == "-arch"; }),
            0);

  // Single arch option is preserved.
  Cmd.CommandLine = {"clang", "-arch", "foo", "/Users/foo.cc"};
  Mangler(Cmd, "/Users/foo.cc");
  EXPECT_EQ(llvm::count_if(Cmd.CommandLine,
                           [](llvm::StringRef Arg) { return Arg == "-arch"; }),
            1);
}

TEST(CommandMangler, EmptyArgs) {
  const auto Mangler = CommandMangler::forTests();
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {};
  // Make sure we don't crash.
  Mangler(Cmd, "foo.cc");
}

TEST(CommandMangler, PathsAsPositional) {
  const auto Mangler = CommandMangler::forTests();
  tooling::CompileCommand Cmd;
  Cmd.CommandLine = {
      "clang",
      "--driver-mode=cl",
      "-I",
      "foo",
  };
  // Make sure we don't crash.
  Mangler(Cmd, "a.cc");
  EXPECT_THAT(Cmd.CommandLine, Contains("foo"));
}
} // namespace
} // namespace clangd
} // namespace clang
