//===- unittests/Interpreter/DeviceOffloadTest.cpp - CUDA device tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the Incremental CUDA compilation infrastructure under Clang's
// Interpreter library. The tests in this file only require the NVPTX backend
// and run without a CUDA toolkit/GPU.
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdlib>

using namespace clang;

namespace {

class DeviceOffloadTest : public InterpreterTestBase {
protected:
  static void SetUpTestSuite() {
    InterpreterTestBase::SetUpTestSuite();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
  }

  void SetUp() override {
    InterpreterTestBase::SetUp();
    if (IsSkipped())
      return;
    std::string Err;
    if (!llvm::TargetRegistry::lookupTarget(llvm::Triple("nvptx64-nvidia-cuda"),
                                            Err))
      GTEST_SKIP() << Err;
  }
};

TEST_F(DeviceOffloadTest, FirstDeviceModuleVerifies) {
#if GTEST_HAS_DEATH_TEST
  // Release builds leave the IR verifier off. A broken module ends in
  // report_fatal_error, so the first device PTU is built in a child process.
  EXPECT_EXIT(
      {
        // Without the runtime headers and libdevice no CUDA toolkit is needed.
        IncrementalCompilerBuilder CB;
        CB.SetCompilerArgs(
            {"-nocudainc", "-nocudalib", "-fverify-intermediate-code"});
        auto DeviceCI = llvm::cantFail(CB.CreateCudaDevice());
        auto HostCI = llvm::cantFail(CB.CreateCudaHost());
        auto Interp = llvm::cantFail(Interpreter::createWithCUDA(
            std::move(HostCI), std::move(DeviceCI)));
        llvm::cantFail(Interp->Parse("__attribute__((device)) void f() {}"));
        exit(0);
      },
      ::testing::ExitedWithCode(0), "");
#else
  GTEST_SKIP() << "no death tests on this platform";
#endif
}

TEST_F(DeviceOffloadTest, EmptyDeviceModule) {
  // Without the runtime headers and libdevice no CUDA toolkit is needed.
  IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs({"-nocudainc", "-nocudalib"});
  auto DeviceCI = CB.CreateCudaDevice();
  ASSERT_THAT_EXPECTED(DeviceCI, llvm::Succeeded());
  auto HostCI = CB.CreateCudaHost();
  ASSERT_THAT_EXPECTED(HostCI, llvm::Succeeded());
  auto Interp =
      Interpreter::createWithCUDA(std::move(*HostCI), std::move(*DeviceCI));
  ASSERT_THAT_EXPECTED(Interp, llvm::Succeeded());

  // A host-only input leaves the device module without a function. Its PTX
  // must still be emitted and handed to the host side.
  auto PTU = (*Interp)->Parse("int i = 0;");
  ASSERT_THAT_EXPECTED(PTU, llvm::Succeeded());

  const CompilerInstance *CI = (*Interp)->getCompilerInstance();
  llvm::StringRef Fatbin = CI->getCodeGenOpts().OffloadBinaryToEmbedFile;
  ASSERT_FALSE(Fatbin.empty());
  auto Buf = CI->getVirtualFileSystem().getBufferForFile(
      Fatbin, /*FileSize=*/-1, /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(static_cast<bool>(Buf)) << Buf.getError().message();
  EXPECT_TRUE((*Buf)->getBuffer().contains(".target"));
}

} // end anonymous namespace
