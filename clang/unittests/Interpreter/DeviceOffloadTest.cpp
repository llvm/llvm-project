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

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
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

class DeviceOffloadHIPTest : public InterpreterTestBase {
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
    if (!llvm::TargetRegistry::lookupTarget(llvm::Triple("amdgcn-amd-amdhsa"),
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
        auto DeviceCI = llvm::cantFail(CB.CreateDevice(OffloadType::CUDA));
        auto HostCI = llvm::cantFail(CB.CreateHost(OffloadType::CUDA));
        auto Interp = llvm::cantFail(Interpreter::createWithDevice(
            std::move(HostCI), std::move(DeviceCI), OffloadType::CUDA));
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
  auto DeviceCI = CB.CreateDevice(OffloadType::CUDA);
  ASSERT_THAT_EXPECTED(DeviceCI, llvm::Succeeded());
  auto HostCI = CB.CreateHost(OffloadType::CUDA);
  ASSERT_THAT_EXPECTED(HostCI, llvm::Succeeded());
  auto Interp = Interpreter::createWithDevice(
      std::move(*HostCI), std::move(*DeviceCI), OffloadType::CUDA);
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

TEST(DeviceOffloadBuilderTest, MultipleOffloadArchsAreRejected) {
  // The driver creates one compile job per architecture. Only one can be used,
  // so this must be reported instead of silently using the first one.
  IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs({"--offload-arch=gfx1100", "--offload-arch=gfx90a",
                      "-nogpulib", "-nogpuinc"});
  auto DeviceCI = CB.CreateDevice(OffloadType::HIP);
  EXPECT_THAT_EXPECTED(
      DeviceCI,
      llvm::FailedWithMessage("Multiple device architectures are not "
                              "supported. Pass a single --offload-arch"));
}

TEST_F(DeviceOffloadHIPTest, DeviceObjectIsGenerated) {
  // The device object is linked into an HSACO with ld.lld.
  std::string Exe = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (!llvm::sys::findProgramByName("ld.lld",
                                    {llvm::sys::path::parent_path(Exe)}) &&
      !llvm::sys::findProgramByName("ld.lld"))
    GTEST_SKIP() << "ld.lld not found";

  IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs({"--offload-arch=gfx1100", "-nogpulib", "-nogpuinc"});
  auto DeviceCI = CB.CreateDevice(OffloadType::HIP);
  ASSERT_THAT_EXPECTED(DeviceCI, llvm::Succeeded());
  auto HostCI = CB.CreateHost(OffloadType::HIP);
  ASSERT_THAT_EXPECTED(HostCI, llvm::Succeeded());
  auto Interp = Interpreter::createWithDevice(
      std::move(*HostCI), std::move(*DeviceCI), OffloadType::HIP);
  ASSERT_THAT_EXPECTED(Interp, llvm::Succeeded());

  // FIXME: Replace this with a check of the embedded bundle once the offload
  // binary is generated.
  auto PTU = (*Interp)->Parse("__attribute__((device)) void f() {}");
  EXPECT_THAT_EXPECTED(
      PTU, llvm::FailedWithMessage("HIP offload binary generation is not "
                                   "supported yet."));
}

// A bitcode library with one float(float) function that doubles its argument,
// or an empty one when Name is empty.
static void writeLibrary(int FD, llvm::StringRef Name) {
  llvm::LLVMContext Ctx;
  llvm::Module Lib("lib", Ctx);
  Lib.setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
  Lib.setDataLayout(Lib.getTargetTriple().computeDataLayout());
  if (!Name.empty()) {
    llvm::Type *FloatTy = llvm::Type::getFloatTy(Ctx);
    llvm::Function *Fn = llvm::Function::Create(
        llvm::FunctionType::get(FloatTy, {FloatTy}, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, Name, Lib);
    llvm::IRBuilder<> Builder(llvm::BasicBlock::Create(Ctx, "", Fn));
    Builder.CreateRet(Builder.CreateFAdd(Fn->getArg(0), Fn->getArg(0)));
  }
  llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
  llvm::WriteBitcodeToFile(Lib, OS);
}

TEST_F(DeviceOffloadTest, BuiltinBitcodeLinkedIntoEveryModule) {
  // Builtin bitcode is linked into every PTU and plain bitcode only once.
  llvm::SmallString<128> BuiltinPath, PlainPath;
  int BuiltinFD, PlainFD;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("builtin", "bc", BuiltinFD,
                                                  BuiltinPath));
  llvm::FileRemover BuiltinRemover(BuiltinPath);
  writeLibrary(BuiltinFD, "twice");
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("plain", "bc", PlainFD, PlainPath));
  llvm::FileRemover PlainRemover(PlainPath);
  writeLibrary(PlainFD, /*Name=*/"");

  IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs({"-nocudainc", "-nocudalib"});
  auto DeviceCI = CB.CreateDevice(OffloadType::CUDA);
  ASSERT_THAT_EXPECTED(DeviceCI, llvm::Succeeded());
  // The entries the driver adds for -mlink-builtin-bitcode and for
  // -mlink-bitcode-file.
  auto &Files = (*DeviceCI)->getCodeGenOpts().LinkBitcodeFiles;
  Files.push_back({std::string(BuiltinPath), /*PropagateAttrs=*/true,
                   /*Internalize=*/true, llvm::Linker::Flags::LinkOnlyNeeded});
  Files.push_back({std::string(PlainPath), /*PropagateAttrs=*/false,
                   /*Internalize=*/false, llvm::Linker::Flags::None});
  auto HostCI = CB.CreateHost(OffloadType::CUDA);
  ASSERT_THAT_EXPECTED(HostCI, llvm::Succeeded());
  auto Interp = Interpreter::createWithDevice(
      std::move(*HostCI), std::move(*DeviceCI), OffloadType::CUDA);
  ASSERT_THAT_EXPECTED(Interp, llvm::Succeeded());
  // The plain file is linked once, into the initial module, and never
  // reopened.
  ASSERT_FALSE(llvm::sys::fs::remove(PlainPath));

  // Every device module that calls twice() must define it.
  for (const char *Code :
       {"extern \"C\" __attribute__((device)) float twice(float);"
        "__attribute__((device)) float f(float x) { return twice(x); }",
        "__attribute__((device)) float g(float x) { return twice(x); }"}) {
    auto PTU = (*Interp)->Parse(Code);
    ASSERT_THAT_EXPECTED(PTU, llvm::Succeeded());
    const CompilerInstance *CI = (*Interp)->getCompilerInstance();
    auto Buf = CI->getVirtualFileSystem().getBufferForFile(
        CI->getCodeGenOpts().OffloadBinaryToEmbedFile, /*FileSize=*/-1,
        /*RequiresNullTerminator=*/false);
    ASSERT_TRUE(static_cast<bool>(Buf)) << Buf.getError().message();
    llvm::StringRef PTX = (*Buf)->getBuffer();
    EXPECT_FALSE(PTX.contains(".extern")) << PTX;
    EXPECT_TRUE(PTX.contains("twice")) << PTX;
  }
}

} // end anonymous namespace
