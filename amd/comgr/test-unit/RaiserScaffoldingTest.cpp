//===- RaiserScaffoldingTest.cpp - Hotswap transpiler scaffolding test ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the scaffolding contract `raiseToIR` advertises: an empty input
// produces a well-formed `llvm::Module` containing one `AMDGPU_KERNEL`
// function whose body is exactly `ret void`, with the AMDGPU triple set.
// Empty inputs succeed; missing kernel descriptor / malformed ISA inputs
// are rejected with a structured failure.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

COMGR::hotswap::KernelMeta makeKernelMeta(llvm::StringRef Name) {
  COMGR::hotswap::KernelMeta Meta;
  Meta.Name = Name.str();
  Meta.HasKernelDescriptor = true;
  return Meta;
}

} // namespace

TEST(RaiserScaffolding, EmptyInputProducesValidModule) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  COMGR::hotswap::RaiseResult Result =
      COMGR::hotswap::raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(Result.Success);
  ASSERT_NE(Result.Ctx, nullptr);
  ASSERT_NE(Result.Module, nullptr);

  std::string Err;
  llvm::raw_string_ostream ErrStream(Err);
  EXPECT_FALSE(llvm::verifyModule(*Result.Module, &ErrStream)) << Err;
}

TEST(RaiserScaffolding, ModuleAdvertisesAMDGPUTriple) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  COMGR::hotswap::RaiseResult Result =
      COMGR::hotswap::raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(Result.Success);
  ASSERT_NE(Result.Module, nullptr);
  EXPECT_EQ(Result.Module->getTargetTriple().str(), "amdgcn-amd-amdhsa");
}

TEST(RaiserScaffolding, KernelFunctionIsAMDGPUKernelWithRetVoid) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  COMGR::hotswap::RaiseResult Result =
      COMGR::hotswap::raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(Result.Success);
  llvm::Function *Fn = Result.Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_EQ(Fn->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);
  ASSERT_EQ(Fn->size(), 1u);
  llvm::BasicBlock &Entry = Fn->getEntryBlock();
  ASSERT_FALSE(Entry.empty());
  EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(Entry.getTerminator()));
}

TEST(RaiserScaffolding, MissingKernelDescriptorIsRejected) {
  COMGR::hotswap::KernelMeta Meta;
  Meta.Name = "kernel";
  Meta.HasKernelDescriptor = false;
  COMGR::hotswap::RaiseResult Result =
      COMGR::hotswap::raiseToIR("gfx942", "kernel", Meta);

  EXPECT_FALSE(Result.Success);
  EXPECT_TRUE(Result.Failure.hasFailed());
}

TEST(RaiserScaffolding, EmptyTargetIsaIsRejected) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  COMGR::hotswap::RaiseResult Result =
      COMGR::hotswap::raiseToIR("", "kernel", Meta);

  EXPECT_FALSE(Result.Success);
  EXPECT_TRUE(Result.Failure.hasFailed());
}

TEST(RaiserScaffolding, MalformedTargetIsaIsRejected) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  COMGR::hotswap::RaiseResult Result =
      COMGR::hotswap::raiseToIR("not-a-real-isa", "kernel", Meta);

  EXPECT_FALSE(Result.Success);
  EXPECT_TRUE(Result.Failure.hasFailed());
}
