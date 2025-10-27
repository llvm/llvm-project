//===- unittests/Driver/DXCModeTest.cpp --- DXC Mode tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for driver DXCMode.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>

#include "SimpleDiagnosticConsumer.h"

using namespace clang;
using namespace clang::driver;

static void validateTargetProfile(
    StringRef TargetProfile, StringRef ExpectTriple,
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> &InMemoryFileSystem,
    DiagnosticsEngine &Diags) {
  Driver TheDriver("/bin/clang", "", Diags, "", InMemoryFileSystem);
  std::unique_ptr<Compilation> C{TheDriver.BuildCompilation(
      {"clang", "--driver-mode=dxc", TargetProfile.data(), "foo.hlsl", "-Vd"})};
  EXPECT_TRUE(C);
  EXPECT_STREQ(TheDriver.getTargetTriple().c_str(), ExpectTriple.data());
  EXPECT_EQ(Diags.getNumErrors(), 0u);
}

static void validateTargetProfile(
    StringRef TargetProfile, StringRef ExpectError,
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> &InMemoryFileSystem,
    DiagnosticsEngine &Diags, SimpleDiagnosticConsumer *DiagConsumer,
    unsigned NumOfErrors) {
  Driver TheDriver("/bin/clang", "", Diags, "", InMemoryFileSystem);
  std::unique_ptr<Compilation> C{TheDriver.BuildCompilation(
      {"clang", "--driver-mode=dxc", TargetProfile.data(), "foo.hlsl", "-Vd"})};
  EXPECT_TRUE(C);
  EXPECT_EQ(Diags.getNumErrors(), NumOfErrors);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(), ExpectError.data());
  DiagConsumer->clear();
}

TEST(DxcModeTest, TargetProfileValidation) {
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  InMemoryFileSystem->addFile("foo.hlsl", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));

  auto *DiagConsumer = new SimpleDiagnosticConsumer;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts, DiagConsumer);

  validateTargetProfile("-Tvs_6_0", "dxilv1.0--shadermodel6.0-vertex",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Ths_6_1", "dxilv1.1--shadermodel6.1-hull",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tds_6_2", "dxilv1.2--shadermodel6.2-domain",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tds_6_2", "dxilv1.2--shadermodel6.2-domain",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tgs_6_3", "dxilv1.3--shadermodel6.3-geometry",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tps_6_4", "dxilv1.4--shadermodel6.4-pixel",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tcs_6_5", "dxilv1.5--shadermodel6.5-compute",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tms_6_6", "dxilv1.6--shadermodel6.6-mesh",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tas_6_7", "dxilv1.7--shadermodel6.7-amplification",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tcs_6_8", "dxilv1.8--shadermodel6.8-compute",
                        InMemoryFileSystem, Diags);
  validateTargetProfile("-Tlib_6_x", "dxilv1.8--shadermodel6.15-library",
                        InMemoryFileSystem, Diags);

  // Invalid tests.
  validateTargetProfile("-Tpss_6_1", "invalid profile : pss_6_1",
                        InMemoryFileSystem, Diags, DiagConsumer, 1);

  validateTargetProfile("-Tps_6_x", "invalid profile : ps_6_x",
                        InMemoryFileSystem, Diags, DiagConsumer, 2);
  validateTargetProfile("-Tlib_6_1", "invalid profile : lib_6_1",
                        InMemoryFileSystem, Diags, DiagConsumer, 3);
  validateTargetProfile("-Tfoo", "invalid profile : foo", InMemoryFileSystem,
                        Diags, DiagConsumer, 4);
  validateTargetProfile("", "target profile option (-T) is missing",
                        InMemoryFileSystem, Diags, DiagConsumer, 5);
}

TEST(DxcModeTest, ValidatorVersionValidation) {
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  InMemoryFileSystem->addFile("foo.hlsl", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));

  auto *DiagConsumer = new SimpleDiagnosticConsumer;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts, DiagConsumer);
  Driver TheDriver("/bin/clang", "", Diags, "", InMemoryFileSystem);
  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(
      {"clang", "--driver-mode=dxc", "-Tlib_6_7", "foo.hlsl"}));
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

  std::unique_ptr<llvm::opt::DerivedArgList> TranslatedArgs{
      TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None)};
  EXPECT_NE(TranslatedArgs, nullptr);
  if (TranslatedArgs) {
    auto *A = TranslatedArgs->getLastArg(
        clang::driver::options::OPT_dxil_validator_version);
    EXPECT_NE(A, nullptr);
    if (A) {
      EXPECT_STREQ(A->getValue(), "1.1");
    }
  }
  EXPECT_EQ(Diags.getNumErrors(), 0u);

  // Invalid tests.
  Args = TheDriver.ParseArgStrings({"-validator-version", "0.1"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs.reset(
      TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None));
  EXPECT_EQ(Diags.getNumErrors(), 1u);
  EXPECT_STREQ(
      DiagConsumer->Errors.back().c_str(),
      "invalid validator version : 0.1; if validator major version is 0, "
      "minor version must also be 0");
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-validator-version", "1"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs.reset(
      TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None));
  EXPECT_EQ(Diags.getNumErrors(), 2u);
  EXPECT_STREQ(DiagConsumer->Errors.back().c_str(),
               "invalid validator version : 1; format of validator version is "
               "\"<major>.<minor>\" (ex:\"1.4\")");
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-validator-version", "-Tlib_6_7"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs.reset(
      TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None));
  EXPECT_EQ(Diags.getNumErrors(), 3u);
  EXPECT_STREQ(
      DiagConsumer->Errors.back().c_str(),
      "invalid validator version : -Tlib_6_7; format of validator version is "
      "\"<major>.<minor>\" (ex:\"1.4\")");
  DiagConsumer->clear();

  Args = TheDriver.ParseArgStrings({"-validator-version", "foo"}, false,
                                   ContainsError);
  EXPECT_FALSE(ContainsError);
  DAL = std::make_unique<llvm::opt::DerivedArgList>(Args);
  for (auto *A : Args)
    DAL->append(A);

  TranslatedArgs.reset(
      TC.TranslateArgs(*DAL, "0", Action::OffloadKind::OFK_None));
  EXPECT_EQ(Diags.getNumErrors(), 4u);
  EXPECT_STREQ(
      DiagConsumer->Errors.back().c_str(),
      "invalid validator version : foo; format of validator version is "
      "\"<major>.<minor>\" (ex:\"1.4\")");
  DiagConsumer->clear();
}

TEST(DxcModeTest, DefaultEntry) {
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);

  InMemoryFileSystem->addFile("foo.hlsl", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));

  const char *Args[] = {"clang", "--driver-mode=dxc", "-Tcs_6_7", "foo.hlsl"};

  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*InMemoryFileSystem, DiagOpts);

  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  std::unique_ptr<CompilerInvocation> CInvok =
      createInvocation(Args, std::move(CIOpts));
  EXPECT_TRUE(CInvok);
  // Make sure default entry is "main".
  EXPECT_STREQ(CInvok->getTargetOpts().HLSLEntry.c_str(), "main");

  const char *EntryArgs[] = {"clang", "--driver-mode=dxc", "-Ebar", "-Tcs_6_7",
                             "foo.hlsl"};
  CInvok = createInvocation(EntryArgs, std::move(CIOpts));
  EXPECT_TRUE(CInvok);
  // Make sure "-E" will set entry.
  EXPECT_STREQ(CInvok->getTargetOpts().HLSLEntry.c_str(), "bar");
}
