//===- unittests/Driver/ToolChainTest.cpp --- ToolChain tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for ToolChains.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/ToolChain.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

#include "SimpleDiagnosticConsumer.h"

using namespace clang;
using namespace clang::driver;

namespace {

TEST(ToolChainTest, VFSGCCInstallation) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  const char *EmptyFiles[] = {
      "foo.cpp",
      "/bin/clang",
      "/usr/lib/gcc/arm-linux-gnueabi/4.6.1/crtbegin.o",
      "/usr/lib/gcc/arm-linux-gnueabi/4.6.1/crtend.o",
      "/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/crtbegin.o",
      "/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/crtend.o",
      "/usr/lib/arm-linux-gnueabi/crt1.o",
      "/usr/lib/arm-linux-gnueabi/crti.o",
      "/usr/lib/arm-linux-gnueabi/crtn.o",
      "/usr/lib/arm-linux-gnueabihf/crt1.o",
      "/usr/lib/arm-linux-gnueabihf/crti.o",
      "/usr/lib/arm-linux-gnueabihf/crtn.o",
      "/usr/include/arm-linux-gnueabi/.keep",
      "/usr/include/arm-linux-gnueabihf/.keep",
      "/lib/arm-linux-gnueabi/.keep",
      "/lib/arm-linux-gnueabihf/.keep",

      "/sysroot/usr/lib/gcc/arm-linux-gnueabi/4.5.1/crtbegin.o",
      "/sysroot/usr/lib/gcc/arm-linux-gnueabi/4.5.1/crtend.o",
      "/sysroot/usr/lib/gcc/arm-linux-gnueabihf/4.5.3/crtbegin.o",
      "/sysroot/usr/lib/gcc/arm-linux-gnueabihf/4.5.3/crtend.o",
      "/sysroot/usr/lib/arm-linux-gnueabi/crt1.o",
      "/sysroot/usr/lib/arm-linux-gnueabi/crti.o",
      "/sysroot/usr/lib/arm-linux-gnueabi/crtn.o",
      "/sysroot/usr/lib/arm-linux-gnueabihf/crt1.o",
      "/sysroot/usr/lib/arm-linux-gnueabihf/crti.o",
      "/sysroot/usr/lib/arm-linux-gnueabihf/crtn.o",
      "/sysroot/usr/include/arm-linux-gnueabi/.keep",
      "/sysroot/usr/include/arm-linux-gnueabihf/.keep",
      "/sysroot/lib/arm-linux-gnueabi/.keep",
      "/sysroot/lib/arm-linux-gnueabihf/.keep",
  };

  for (const char *Path : EmptyFiles)
    InMemoryFileSystem->addFile(Path, 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "arm-linux-gnueabihf", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"-fsyntax-only", "--gcc-toolchain=", "--sysroot=", "foo.cpp"}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ(
        "Found candidate GCC installation: "
        "/usr/lib/gcc/arm-linux-gnueabihf/4.6.3\n"
        "Selected GCC installation: /usr/lib/gcc/arm-linux-gnueabihf/4.6.3\n"
        "Candidate multilib: .;@m32\n"
        "Selected multilib: .;@m32\n",
        S);
  }

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "arm-linux-gnueabihf", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"-fsyntax-only", "--gcc-toolchain=", "--sysroot=/sysroot",
         "foo.cpp"}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    // Test that 4.5.3 from --sysroot is not overridden by 4.6.3 (larger
    // version) from /usr.
    EXPECT_EQ("Found candidate GCC installation: "
              "/sysroot/usr/lib/gcc/arm-linux-gnueabihf/4.5.3\n"
              "Selected GCC installation: "
              "/sysroot/usr/lib/gcc/arm-linux-gnueabihf/4.5.3\n"
              "Candidate multilib: .;@m32\n"
              "Selected multilib: .;@m32\n",
              S);
  }
}

TEST(ToolChainTest, VFSGCCInstallationRelativeDir) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                   "clang LLVM compiler", InMemoryFileSystem);

  const char *EmptyFiles[] = {
      "foo.cpp", "/home/test/lib/gcc/arm-linux-gnueabi/4.6.1/crtbegin.o",
      "/home/test/include/arm-linux-gnueabi/.keep"};

  for (const char *Path : EmptyFiles)
    InMemoryFileSystem->addFile(Path, 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
      {"-fsyntax-only", "--gcc-toolchain=", "foo.cpp"}));
  EXPECT_TRUE(C);

  std::string S;
  {
    llvm::raw_string_ostream OS(S);
    C->getDefaultToolChain().printVerboseInfo(OS);
  }
  if (is_style_windows(llvm::sys::path::Style::native))
    std::replace(S.begin(), S.end(), '\\', '/');
  EXPECT_EQ("Found candidate GCC installation: "
            "/home/test/bin/../lib/gcc/arm-linux-gnueabi/4.6.1\n"
            "Selected GCC installation: "
            "/home/test/bin/../lib/gcc/arm-linux-gnueabi/4.6.1\n"
            "Candidate multilib: .;@m32\n"
            "Selected multilib: .;@m32\n",
            S);
}

TEST(ToolChainTest, VFSSolarisMultiGCCInstallation) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  const char *EmptyFiles[] = {
      // Sort entries so the latest version doesn't come first.
      "/usr/gcc/7/lib/gcc/sparcv9-sun-solaris2.11/7.5.0/32/crtbegin.o",
      "/usr/gcc/7/lib/gcc/sparcv9-sun-solaris2.11/7.5.0/crtbegin.o",
      "/usr/gcc/7/lib/gcc/x86_64-pc-solaris2.11/7.5.0/32/crtbegin.o",
      "/usr/gcc/7/lib/gcc/x86_64-pc-solaris2.11/7.5.0/crtbegin.o",
      "/usr/gcc/11/lib/gcc/sparcv9-sun-solaris2.11/11.4.0/crtbegin.o",
      "/usr/gcc/11/lib/gcc/sparcv9-sun-solaris2.11/11.4.0/sparcv8plus/crtbegin.o",
      "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0/32/crtbegin.o",
      "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0/crtbegin.o",
      "/usr/gcc/4.7/lib/gcc/i386-pc-solaris2.11/4.7.3/amd64/crtbegin.o",
      "/usr/gcc/4.7/lib/gcc/i386-pc-solaris2.11/4.7.3/crtbegin.o",
      "/usr/gcc/4.7/lib/gcc/sparc-sun-solaris2.11/4.7.3/crtbegin.o",
      "/usr/gcc/4.7/lib/gcc/sparc-sun-solaris2.11/4.7.3/sparcv9/crtbegin.o",
  };

  for (const char *Path : EmptyFiles)
    InMemoryFileSystem->addFile(Path, 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "i386-pc-solaris2.11", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(
        TheDriver.BuildCompilation({"-v", "--gcc-toolchain=", "--sysroot="}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0\n"
              "Selected GCC installation: "
              "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0\n"
              "Candidate multilib: .;@m64\n"
              "Candidate multilib: 32;@m32\n"
              "Selected multilib: 32;@m32\n",
              S);
  }

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "amd64-pc-solaris2.11", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(
        TheDriver.BuildCompilation({"-v", "--gcc-toolchain=", "--sysroot="}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0\n"
              "Selected GCC installation: "
              "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0\n"
              "Candidate multilib: .;@m64\n"
              "Candidate multilib: 32;@m32\n"
              "Selected multilib: .;@m64\n",
              S);
  }

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "x86_64-pc-solaris2.11", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(
        TheDriver.BuildCompilation({"-v", "--gcc-toolchain=", "--sysroot="}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0\n"
              "Selected GCC installation: "
              "/usr/gcc/11/lib/gcc/x86_64-pc-solaris2.11/11.4.0\n"
              "Candidate multilib: .;@m64\n"
              "Candidate multilib: 32;@m32\n"
              "Selected multilib: .;@m64\n",
              S);
  }

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "sparc-sun-solaris2.11", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(
        TheDriver.BuildCompilation({"-v", "--gcc-toolchain=", "--sysroot="}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/usr/gcc/11/lib/gcc/sparcv9-sun-solaris2.11/11.4.0\n"
              "Selected GCC installation: "
              "/usr/gcc/11/lib/gcc/sparcv9-sun-solaris2.11/11.4.0\n"
              "Candidate multilib: .;@m64\n"
              "Candidate multilib: sparcv8plus;@m32\n"
              "Selected multilib: sparcv8plus;@m32\n",
              S);
  }
  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "sparcv9-sun-solaris2.11", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(
        TheDriver.BuildCompilation({"-v", "--gcc-toolchain=", "--sysroot="}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/usr/gcc/11/lib/gcc/sparcv9-sun-solaris2.11/11.4.0\n"
              "Selected GCC installation: "
              "/usr/gcc/11/lib/gcc/sparcv9-sun-solaris2.11/11.4.0\n"
              "Candidate multilib: .;@m64\n"
              "Candidate multilib: sparcv8plus;@m32\n"
              "Selected multilib: .;@m64\n",
              S);
  }
}

MATCHER_P(jobHasArgs, Substr, "") {
  const driver::Command &C = arg;
  std::string Args = "";
  llvm::ListSeparator Sep(" ");
  for (const char *Arg : C.getArguments()) {
    Args += Sep;
    Args += Arg;
  }
  if (is_style_windows(llvm::sys::path::Style::native))
    std::replace(Args.begin(), Args.end(), '\\', '/');
  if (llvm::StringRef(Args).contains(Substr))
    return true;
  *result_listener << "whose args are '" << Args << "'";
  return false;
}

TEST(ToolChainTest, VFSGnuLibcxxPathNoSysroot) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  const char *EmptyFiles[] = {
      "foo.cpp",
      "/bin/clang",
      "/usr/include/c++/v1/cstdio",
  };

  for (const char *Path : EmptyFiles)
    InMemoryFileSystem->addFile(Path, 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

  {
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "x86_64-unknown-linux-gnu", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/bin/clang", "-fsyntax-only", "-stdlib=libc++",
         "--sysroot=", "foo.cpp"}));
    ASSERT_TRUE(C);
    EXPECT_THAT(C->getJobs(), testing::ElementsAre(jobHasArgs(
                                  "-internal-isystem /usr/include/c++/v1")));
  }
}

TEST(ToolChainTest, DefaultDriverMode) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  Driver CCDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                  "clang LLVM compiler", InMemoryFileSystem);
  CCDriver.setCheckInputsExist(false);
  Driver CXXDriver("/home/test/bin/clang++", "arm-linux-gnueabi", Diags,
                   "clang LLVM compiler", InMemoryFileSystem);
  CXXDriver.setCheckInputsExist(false);
  Driver CLDriver("/home/test/bin/clang-cl", "arm-linux-gnueabi", Diags,
                  "clang LLVM compiler", InMemoryFileSystem);
  CLDriver.setCheckInputsExist(false);

  std::unique_ptr<Compilation> CC(CCDriver.BuildCompilation(
      { "/home/test/bin/clang", "foo.cpp"}));
  std::unique_ptr<Compilation> CXX(CXXDriver.BuildCompilation(
      { "/home/test/bin/clang++", "foo.cpp"}));
  std::unique_ptr<Compilation> CL(CLDriver.BuildCompilation(
      { "/home/test/bin/clang-cl", "foo.cpp"}));

  EXPECT_TRUE(CC);
  EXPECT_TRUE(CXX);
  EXPECT_TRUE(CL);
  EXPECT_TRUE(CCDriver.CCCIsCC());
  EXPECT_TRUE(CXXDriver.CCCIsCXX());
  EXPECT_TRUE(CLDriver.IsCLMode());
}
TEST(ToolChainTest, InvalidArgument) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  Driver TheDriver("/bin/clang", "arm-linux-gnueabihf", Diags);
  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
      {"-fsyntax-only", "-fan-unknown-option", "foo.cpp"}));
  EXPECT_TRUE(C);
  EXPECT_TRUE(C->containsError());
}

TEST(ToolChainTest, ParsedClangName) {
  ParsedClangName Empty;
  EXPECT_TRUE(Empty.TargetPrefix.empty());
  EXPECT_TRUE(Empty.ModeSuffix.empty());
  EXPECT_TRUE(Empty.DriverMode == nullptr);
  EXPECT_FALSE(Empty.TargetIsValid);

  ParsedClangName DriverOnly("clang", nullptr);
  EXPECT_TRUE(DriverOnly.TargetPrefix.empty());
  EXPECT_TRUE(DriverOnly.ModeSuffix == "clang");
  EXPECT_TRUE(DriverOnly.DriverMode == nullptr);
  EXPECT_FALSE(DriverOnly.TargetIsValid);

  ParsedClangName DriverOnly2("clang++", "--driver-mode=g++");
  EXPECT_TRUE(DriverOnly2.TargetPrefix.empty());
  EXPECT_TRUE(DriverOnly2.ModeSuffix == "clang++");
  EXPECT_STREQ(DriverOnly2.DriverMode, "--driver-mode=g++");
  EXPECT_FALSE(DriverOnly2.TargetIsValid);

  ParsedClangName TargetAndMode("i386", "clang-g++", "--driver-mode=g++", true);
  EXPECT_TRUE(TargetAndMode.TargetPrefix == "i386");
  EXPECT_TRUE(TargetAndMode.ModeSuffix == "clang-g++");
  EXPECT_STREQ(TargetAndMode.DriverMode, "--driver-mode=g++");
  EXPECT_TRUE(TargetAndMode.TargetIsValid);
}

TEST(ToolChainTest, GetTargetAndMode) {
  llvm::InitializeAllTargets();
  std::string IgnoredError;
  if (!llvm::TargetRegistry::lookupTarget("x86_64", IgnoredError))
    GTEST_SKIP();

  ParsedClangName Res = ToolChain::getTargetAndModeFromProgramName("clang");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix == "clang");
  EXPECT_TRUE(Res.DriverMode == nullptr);
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("clang++");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix == "clang++");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=g++");
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("clang++6.0");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix == "clang++");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=g++");
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("clang++-release");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix == "clang++");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=g++");
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("x86_64-clang++");
  EXPECT_TRUE(Res.TargetPrefix == "x86_64");
  EXPECT_TRUE(Res.ModeSuffix == "clang++");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=g++");
  EXPECT_TRUE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName(
      "x86_64-linux-gnu-clang-c++");
  EXPECT_TRUE(Res.TargetPrefix == "x86_64-linux-gnu");
  EXPECT_TRUE(Res.ModeSuffix == "clang-c++");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=g++");
  EXPECT_TRUE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName(
      "x86_64-linux-gnu-clang-c++-tot");
  EXPECT_TRUE(Res.TargetPrefix == "x86_64-linux-gnu");
  EXPECT_TRUE(Res.ModeSuffix == "clang-c++");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=g++");
  EXPECT_TRUE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("qqq");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix.empty());
  EXPECT_TRUE(Res.DriverMode == nullptr);
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("x86_64-qqq");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix.empty());
  EXPECT_TRUE(Res.DriverMode == nullptr);
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("qqq-clang-cl");
  EXPECT_TRUE(Res.TargetPrefix == "qqq");
  EXPECT_TRUE(Res.ModeSuffix == "clang-cl");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=cl");
  EXPECT_FALSE(Res.TargetIsValid);

  Res = ToolChain::getTargetAndModeFromProgramName("clang-dxc");
  EXPECT_TRUE(Res.TargetPrefix.empty());
  EXPECT_TRUE(Res.ModeSuffix == "clang-dxc");
  EXPECT_STREQ(Res.DriverMode, "--driver-mode=dxc");
  EXPECT_FALSE(Res.TargetIsValid);
}

TEST(ToolChainTest, CommandOutput) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  Driver CCDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                  "clang LLVM compiler", InMemoryFileSystem);
  CCDriver.setCheckInputsExist(false);
  std::unique_ptr<Compilation> CC(
      CCDriver.BuildCompilation({"/home/test/bin/clang", "foo.cpp"}));
  const JobList &Jobs = CC->getJobs();

  const auto &CmdCompile = Jobs.getJobs().front();
  const auto &InFile = CmdCompile->getInputInfos().front().getFilename();
  EXPECT_STREQ(InFile, "foo.cpp");
  auto ObjFile = CmdCompile->getOutputFilenames().front();
  EXPECT_TRUE(StringRef(ObjFile).ends_with(".o"));

  const auto &CmdLink = Jobs.getJobs().back();
  const auto LinkInFile = CmdLink->getInputInfos().front().getFilename();
  EXPECT_EQ(ObjFile, LinkInFile);
  auto ExeFile = CmdLink->getOutputFilenames().front();
  EXPECT_EQ("a.out", ExeFile);
}

TEST(ToolChainTest, PostCallback) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  // The executable path must not exist.
  Driver CCDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                  "clang LLVM compiler", InMemoryFileSystem);
  CCDriver.setCheckInputsExist(false);
  std::unique_ptr<Compilation> CC(
      CCDriver.BuildCompilation({"/home/test/bin/clang", "foo.cpp"}));
  bool CallbackHasCalled = false;
  CC->setPostCallback(
      [&](const Command &C, int Ret) { CallbackHasCalled = true; });
  const JobList &Jobs = CC->getJobs();
  auto &CmdCompile = Jobs.getJobs().front();
  const Command *FailingCmd = nullptr;
  CC->ExecuteCommand(*CmdCompile, FailingCmd);
  EXPECT_TRUE(CallbackHasCalled);
}

TEST(CompilerInvocation, SplitSwarfSingleCrash) {
  static constexpr const char *Args[] = {
      "clang",     "--target=arm-linux-gnueabi",
      "-gdwarf-4", "-gsplit-dwarf=single",
      "-c",        "foo.cpp"};
  CreateInvocationOptions CIOpts;
  std::unique_ptr<CompilerInvocation> CI = createInvocation(Args, CIOpts);
  EXPECT_TRUE(CI); // no-crash
}

TEST(ToolChainTest, UEFICallingConventionTest) {
  clang::CompilerInstance compiler;
  compiler.createDiagnostics();

  std::string TrStr = "x86_64-unknown-uefi";
  llvm::Triple Tr(TrStr);
  Tr.setOS(llvm::Triple::OSType::UEFI);
  Tr.setVendor(llvm::Triple::VendorType::UnknownVendor);
  Tr.setEnvironment(llvm::Triple::EnvironmentType::UnknownEnvironment);
  Tr.setArch(llvm::Triple::ArchType::x86_64);

  compiler.getTargetOpts().Triple = Tr.getTriple();
  compiler.setTarget(clang::TargetInfo::CreateTargetInfo(
      compiler.getDiagnostics(),
      std::make_shared<clang::TargetOptions>(compiler.getTargetOpts())));

  EXPECT_EQ(compiler.getTarget().getCallingConvKind(true),
            TargetInfo::CallingConvKind::CCK_MicrosoftWin64);
}

TEST(GetDriverMode, PrefersLastDriverMode) {
  static constexpr const char *Args[] = {"clang-cl", "--driver-mode=foo",
                                         "--driver-mode=bar", "foo.cpp"};
  EXPECT_EQ(getDriverMode(Args[0], llvm::ArrayRef(Args).slice(1)), "bar");
}

struct SimpleDiagnosticConsumer : public DiagnosticConsumer {
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    if (DiagLevel == DiagnosticsEngine::Level::Error) {
      Errors.emplace_back();
      Info.FormatDiagnostic(Errors.back());
    } else {
      Msgs.emplace_back();
      Info.FormatDiagnostic(Msgs.back());
    }
  }
  void clear() override {
    Msgs.clear();
    Errors.clear();
    DiagnosticConsumer::clear();
  }
  std::vector<SmallString<32>> Msgs;
  std::vector<SmallString<32>> Errors;
};

TEST(ToolChainTest, ConfigFileSearch) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS(
      new llvm::vfs::InMemoryFileSystem);

#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "/";
#endif
  FS->setCurrentWorkingDirectory(TestRoot);

  FS->addFile(
      "/opt/sdk/root.cfg", 0,
      llvm::MemoryBuffer::getMemBuffer("--sysroot=/opt/sdk/platform0\n"));
  FS->addFile(
      "/home/test/sdk/root.cfg", 0,
      llvm::MemoryBuffer::getMemBuffer("--sysroot=/opt/sdk/platform1\n"));
  FS->addFile(
      "/home/test/bin/root.cfg", 0,
      llvm::MemoryBuffer::getMemBuffer("--sysroot=/opt/sdk/platform2\n"));

  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "root.cfg",
         "--config-system-dir=/opt/sdk", "--config-user-dir=/home/test/sdk"}));
    ASSERT_TRUE(C);
    ASSERT_FALSE(C->containsError());
    EXPECT_EQ("/opt/sdk/platform1", TheDriver.SysRoot);
  }
  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "root.cfg",
         "--config-system-dir=/opt/sdk", "--config-user-dir="}));
    ASSERT_TRUE(C);
    ASSERT_FALSE(C->containsError());
    EXPECT_EQ("/opt/sdk/platform0", TheDriver.SysRoot);
  }
  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "root.cfg",
         "--config-system-dir=", "--config-user-dir="}));
    ASSERT_TRUE(C);
    ASSERT_FALSE(C->containsError());
    EXPECT_EQ("/opt/sdk/platform2", TheDriver.SysRoot);
  }
}

struct FileSystemWithError : public llvm::vfs::FileSystem {
  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
    return std::make_error_code(std::errc::no_such_file_or_directory);
  }
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const Twine &Path) override {
    return std::make_error_code(std::errc::permission_denied);
  }
  llvm::vfs::directory_iterator dir_begin(const Twine &Dir,
                                          std::error_code &EC) override {
    return llvm::vfs::directory_iterator();
  }
  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    return std::make_error_code(std::errc::permission_denied);
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return std::make_error_code(std::errc::permission_denied);
  }
};

TEST(ToolChainTest, ConfigFileError) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  std::unique_ptr<SimpleDiagnosticConsumer> DiagConsumer(
      new SimpleDiagnosticConsumer());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagConsumer.get(), false);
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS(new FileSystemWithError);

  Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                   "clang LLVM compiler", FS);
  std::unique_ptr<Compilation> C(
      TheDriver.BuildCompilation({"/home/test/bin/clang", "--no-default-config",
                                  "--config", "./root.cfg", "--version"}));
  ASSERT_TRUE(C);
  ASSERT_TRUE(C->containsError());
  EXPECT_EQ(1U, Diags.getNumErrors());
  EXPECT_STREQ("configuration file './root.cfg' cannot be opened: cannot get "
               "absolute path",
               DiagConsumer->Errors[0].c_str());
}

TEST(ToolChainTest, BadConfigFile) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  std::unique_ptr<SimpleDiagnosticConsumer> DiagConsumer(
      new SimpleDiagnosticConsumer());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagConsumer.get(), false);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS(
      new llvm::vfs::InMemoryFileSystem);

#ifdef _WIN32
  const char *TestRoot = "C:\\";
#define FILENAME "C:/opt/root.cfg"
#define DIRNAME "C:/opt"
#else
  const char *TestRoot = "/";
#define FILENAME "/opt/root.cfg"
#define DIRNAME "/opt"
#endif
  // UTF-16 string must be aligned on 2-byte boundary. Strings and char arrays
  // do not provide necessary alignment, so copy constant string into properly
  // allocated memory in heap.
  llvm::BumpPtrAllocator Alloc;
  char *StrBuff = (char *)Alloc.Allocate(16, 4);
  std::memset(StrBuff, 0, 16);
  std::memcpy(StrBuff, "\xFF\xFE\x00\xD8\x00\x00", 6);
  StringRef BadUTF(StrBuff, 6);
  FS->setCurrentWorkingDirectory(TestRoot);
  FS->addFile("/opt/root.cfg", 0, llvm::MemoryBuffer::getMemBuffer(BadUTF));
  FS->addFile("/home/user/test.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("@file.rsp"));

  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "/opt/root.cfg", "--version"}));
    ASSERT_TRUE(C);
    ASSERT_TRUE(C->containsError());
    EXPECT_EQ(1U, DiagConsumer->Errors.size());
    EXPECT_STREQ("cannot read configuration file '" FILENAME
                 "': Could not convert UTF16 to UTF8",
                 DiagConsumer->Errors[0].c_str());
  }
  DiagConsumer->clear();
  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "/opt", "--version"}));
    ASSERT_TRUE(C);
    ASSERT_TRUE(C->containsError());
    EXPECT_EQ(1U, DiagConsumer->Errors.size());
    EXPECT_STREQ("configuration file '" DIRNAME
                 "' cannot be opened: not a regular file",
                 DiagConsumer->Errors[0].c_str());
  }
  DiagConsumer->clear();
  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "root",
         "--config-system-dir=", "--config-user-dir=", "--version"}));
    ASSERT_TRUE(C);
    ASSERT_TRUE(C->containsError());
    EXPECT_EQ(1U, DiagConsumer->Errors.size());
    EXPECT_STREQ("configuration file 'root' cannot be found",
                 DiagConsumer->Errors[0].c_str());
  }

#undef FILENAME
#undef DIRNAME
}

TEST(ToolChainTest, ConfigInexistentInclude) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  std::unique_ptr<SimpleDiagnosticConsumer> DiagConsumer(
      new SimpleDiagnosticConsumer());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagConsumer.get(), false);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS(
      new llvm::vfs::InMemoryFileSystem);

#ifdef _WIN32
  const char *TestRoot = "C:\\";
#define USERCONFIG "C:\\home\\user\\test.cfg"
#define UNEXISTENT "C:\\home\\user\\file.rsp"
#else
  const char *TestRoot = "/";
#define USERCONFIG "/home/user/test.cfg"
#define UNEXISTENT "/home/user/file.rsp"
#endif
  FS->setCurrentWorkingDirectory(TestRoot);
  FS->addFile("/home/user/test.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("@file.rsp"));

  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "test.cfg",
         "--config-system-dir=", "--config-user-dir=/home/user", "--version"}));
    ASSERT_TRUE(C);
    ASSERT_TRUE(C->containsError());
    EXPECT_EQ(1U, DiagConsumer->Errors.size());
    EXPECT_STRCASEEQ("cannot read configuration file '" USERCONFIG
                     "': cannot not open file '" UNEXISTENT
                     "': no such file or directory",
                     DiagConsumer->Errors[0].c_str());
  }

#undef USERCONFIG
#undef UNEXISTENT
}

TEST(ToolChainTest, ConfigRecursiveInclude) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  std::unique_ptr<SimpleDiagnosticConsumer> DiagConsumer(
      new SimpleDiagnosticConsumer());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagConsumer.get(), false);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS(
      new llvm::vfs::InMemoryFileSystem);

#ifdef _WIN32
  const char *TestRoot = "C:\\";
#define USERCONFIG "C:\\home\\user\\test.cfg"
#define INCLUDED1 "C:\\home\\user\\file1.cfg"
#else
  const char *TestRoot = "/";
#define USERCONFIG "/home/user/test.cfg"
#define INCLUDED1 "/home/user/file1.cfg"
#endif
  FS->setCurrentWorkingDirectory(TestRoot);
  FS->addFile("/home/user/test.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("@file1.cfg"));
  FS->addFile("/home/user/file1.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("@file2.cfg"));
  FS->addFile("/home/user/file2.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("@file3.cfg"));
  FS->addFile("/home/user/file3.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("@file1.cfg"));

  {
    Driver TheDriver("/home/test/bin/clang", "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "test.cfg",
         "--config-system-dir=", "--config-user-dir=/home/user", "--version"}));
    ASSERT_TRUE(C);
    ASSERT_TRUE(C->containsError());
    EXPECT_EQ(1U, DiagConsumer->Errors.size());
    EXPECT_STREQ("cannot read configuration file '" USERCONFIG
                 "': recursive expansion of: '" INCLUDED1 "'",
                 DiagConsumer->Errors[0].c_str());
  }

#undef USERCONFIG
#undef INCLUDED1
}

TEST(ToolChainTest, NestedConfigFile) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS(
      new llvm::vfs::InMemoryFileSystem);

#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "/";
#endif
  FS->setCurrentWorkingDirectory(TestRoot);

  FS->addFile("/opt/sdk/root.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("--config=platform.cfg\n"));
  FS->addFile("/opt/sdk/platform.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("--sysroot=/platform-sys\n"));
  FS->addFile("/home/test/bin/platform.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("--sysroot=/platform-bin\n"));

  SmallString<128> ClangExecutable("/home/test/bin/clang");
  FS->makeAbsolute(ClangExecutable);

  // User file is absent - use system definitions.
  {
    Driver TheDriver(ClangExecutable, "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "root.cfg",
         "--config-system-dir=/opt/sdk", "--config-user-dir=/home/test/sdk"}));
    ASSERT_TRUE(C);
    ASSERT_FALSE(C->containsError());
    EXPECT_EQ("/platform-sys", TheDriver.SysRoot);
  }

  // User file overrides system definitions.
  FS->addFile("/home/test/sdk/platform.cfg", 0,
              llvm::MemoryBuffer::getMemBuffer("--sysroot=/platform-user\n"));
  {
    Driver TheDriver(ClangExecutable, "arm-linux-gnueabi", Diags,
                     "clang LLVM compiler", FS);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
        {"/home/test/bin/clang", "--config", "root.cfg",
         "--config-system-dir=/opt/sdk", "--config-user-dir=/home/test/sdk"}));
    ASSERT_TRUE(C);
    ASSERT_FALSE(C->containsError());
    EXPECT_EQ("/platform-user", TheDriver.SysRoot);
  }
}

} // end anonymous namespace.
