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
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
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
    return;

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
  EXPECT_TRUE(StringRef(ObjFile).endswith(".o"));

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

TEST(GetDriverMode, PrefersLastDriverMode) {
  static constexpr const char *Args[] = {"clang-cl", "--driver-mode=foo",
                                         "--driver-mode=bar", "foo.cpp"};
  EXPECT_EQ(getDriverMode(Args[0], llvm::makeArrayRef(Args).slice(1)), "bar");
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

TEST(DxcModeTest, TargetProfileValidation) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  InMemoryFileSystem->addFile("foo.hlsl", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));

  auto *DiagConsumer = new SimpleDiagnosticConsumer;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagConsumer);
  Driver TheDriver("/bin/clang", "", Diags, "", InMemoryFileSystem);
  std::unique_ptr<Compilation> C(
      TheDriver.BuildCompilation({"clang", "--driver-mode=dxc", "foo.hlsl"}));
  EXPECT_TRUE(C);
  EXPECT_TRUE(!C->containsError());

  auto &TC = C->getDefaultToolChain();
  bool ContainsError = false;
  auto Args = TheDriver.ParseArgStrings({"-Tvs_6_0"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  auto Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.0-vertex");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Ths_6_1"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.1-hull");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tds_6_2"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.2-domain");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tds_6_2"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.2-domain");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tgs_6_3"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.3-geometry");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tps_6_4"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.4-pixel");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tcs_6_5"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.5-compute");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tms_6_6"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.6-mesh");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tas_6_7"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.7-amplification");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  Args = TheDriver.ParseArgStrings({"-Tlib_6_x"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "dxil--shadermodel6.15-library");
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  // Invalid tests.
  Args = TheDriver.ParseArgStrings({"-Tpss_6_1"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "unknown-unknown-shadermodel");
  EXPECT_EQ(Diags.getNumErrors(), 1u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(),
               "invalid profile : pss_6_1");
  Diags.Clear();
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-Tps_6_x"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "unknown-unknown-shadermodel");
  EXPECT_EQ(Diags.getNumErrors(), 2u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(), "invalid profile : ps_6_x");
  Diags.Clear();
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-Tlib_6_1"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "unknown-unknown-shadermodel");
  EXPECT_EQ(Diags.getNumErrors(), 3u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(),
               "invalid profile : lib_6_1");
  Diags.Clear();
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-Tfoo"}, false, ContainsError);
  EXPECT_FALSE(ContainsError);
  Triple = TC.ComputeEffectiveClangTriple(Args);
  EXPECT_STREQ(Triple.c_str(), "unknown-unknown-shadermodel");
  EXPECT_EQ(Diags.getNumErrors(), 4u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(), "invalid profile : foo");
  Diags.Clear();
  DiagConsumer->clear();
}

TEST(DxcModeTest, ValidatorVersionValidation) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  InMemoryFileSystem->addFile("foo.hlsl", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));

  auto *DiagConsumer = new SimpleDiagnosticConsumer;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagConsumer);
  Driver TheDriver("/bin/clang", "", Diags, "", InMemoryFileSystem);
  std::unique_ptr<Compilation> C(
      TheDriver.BuildCompilation({"clang", "--driver-mode=dxc", "foo.hlsl"}));
  EXPECT_TRUE(C);
  EXPECT_TRUE(!C->containsError());

  auto &TC = C->getDefaultToolChain();
  bool ContainsError = false;
  auto Args = TheDriver.ParseArgStrings({"-validator-version", "1.1"}, false,
                                        ContainsError);
  EXPECT_FALSE(ContainsError);
  auto DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  auto *TranslatedArgs =
      TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None);
  EXPECT_NE(TranslatedArgs, nullptr);
  if (TranslatedArgs) {
    auto *A = TranslatedArgs->getLastArg(
        clang::driver::options::OPT_dxil_validator_version);
    EXPECT_NE(A, nullptr);
    if (A)
      EXPECT_STREQ(A->getValue(), "1.1");
  }
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  // Invalid tests.
  Args = TheDriver.ParseArgStrings({"-validator-version", "0.1"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs = TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None);
  EXPECT_EQ(Diags.getNumErrors(), 1u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(),
               "invalid validator version : 0.1\nIf validator major version is "
               "0, minor version must also be 0.");
  Diags.Clear();
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-validator-version", "1"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs = TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None);
  EXPECT_EQ(Diags.getNumErrors(), 2u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(),
               "invalid validator version : 1\nFormat of validator version is "
               "\"<major>.<minor>\" (ex:\"1.4\").");
  Diags.Clear();
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-validator-version", "-Tlib_6_7"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs = TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None);
  EXPECT_EQ(Diags.getNumErrors(), 3u);
  EXPECT_STREQ(
      DiagConsumer->Errors.back().c_str(),
      "invalid validator version : -Tlib_6_7\nFormat of validator version is "
      "\"<major>.<minor>\" (ex:\"1.4\").");
  Diags.Clear();
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-validator-version", "foo"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs = TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None);
  EXPECT_EQ(Diags.getNumErrors(), 4u);
  EXPECT_STREQ(
      DiagConsumer->Errors.back().c_str(),
      "invalid validator version : foo\nFormat of validator version is "
      "\"<major>.<minor>\" (ex:\"1.4\").");
  Diags.Clear();
  DiagConsumer->clear();
}

TEST(ToolChainTest, Toolsets) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};

  // Check (newer) GCC toolset installation.
  {
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
        new llvm::vfs::InMemoryFileSystem);

    // These should be ignored
    InMemoryFileSystem->addFile("/opt/rh/gcc-toolset-2", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));
    InMemoryFileSystem->addFile("/opt/rh/gcc-toolset-", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));
    InMemoryFileSystem->addFile("/opt/rh/gcc-toolset--", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));
    InMemoryFileSystem->addFile("/opt/rh/gcc-toolset--1", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

    // File needed for GCC installation detection.
    InMemoryFileSystem->addFile(
        "/opt/rh/gcc-toolset-12/lib/gcc/x86_64-redhat-linux/11/crtbegin.o", 0,
        llvm::MemoryBuffer::getMemBuffer("\n"));

    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "x86_64-redhat-linux", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation({"-v"}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/opt/rh/gcc-toolset-12/lib/gcc/x86_64-redhat-linux/11\n"
              "Selected GCC installation: "
              "/opt/rh/gcc-toolset-12/lib/gcc/x86_64-redhat-linux/11\n"
              "Candidate multilib: .;@m64\n"
              "Selected multilib: .;@m64\n",
              S);
  }

  // And older devtoolset.
  {
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
        new llvm::vfs::InMemoryFileSystem);

    // These should be ignored
    InMemoryFileSystem->addFile("/opt/rh/devtoolset-2", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));
    InMemoryFileSystem->addFile("/opt/rh/devtoolset-", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));
    InMemoryFileSystem->addFile("/opt/rh/devtoolset--", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));
    InMemoryFileSystem->addFile("/opt/rh/devtoolset--1", 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

    // File needed for GCC installation detection.
    InMemoryFileSystem->addFile(
        "/opt/rh/devtoolset-12/lib/gcc/x86_64-redhat-linux/11/crtbegin.o", 0,
        llvm::MemoryBuffer::getMemBuffer("\n"));

    DiagnosticsEngine Diags(DiagID, &*DiagOpts, new TestDiagnosticConsumer);
    Driver TheDriver("/bin/clang", "x86_64-redhat-linux", Diags,
                     "clang LLVM compiler", InMemoryFileSystem);
    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation({"-v"}));
    ASSERT_TRUE(C);
    std::string S;
    {
      llvm::raw_string_ostream OS(S);
      C->getDefaultToolChain().printVerboseInfo(OS);
    }
    if (is_style_windows(llvm::sys::path::Style::native))
      std::replace(S.begin(), S.end(), '\\', '/');
    EXPECT_EQ("Found candidate GCC installation: "
              "/opt/rh/devtoolset-12/lib/gcc/x86_64-redhat-linux/11\n"
              "Selected GCC installation: "
              "/opt/rh/devtoolset-12/lib/gcc/x86_64-redhat-linux/11\n"
              "Candidate multilib: .;@m64\n"
              "Selected multilib: .;@m64\n",
              S);
  }
}

} // end anonymous namespace.
