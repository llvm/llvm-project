//===- raiser_empty_module_test.cpp - empty-module smoke test -------------===//
//
// Verifies that the bare-bones raiser produces a well-formed empty module:
// a single AMDGPU_KERNEL function whose body is `ret void`.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser.hpp"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace transpiler;

TEST(Raiser, EmptyModuleIsValid) {
  KernelMeta meta;
  meta.name = "kernel";
  meta.hasKernelDescriptor = true;
  RaiseResult result = raiseToIR({}, "gfx942", "kernel", meta);

  ASSERT_TRUE(result.success);
  ASSERT_NE(result.module, nullptr);

  std::string err;
  llvm::raw_string_ostream errStream(err);
  EXPECT_FALSE(llvm::verifyModule(*result.module, &errStream)) << err;

  llvm::Function *fn = result.module->getFunction("kernel");
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);
  EXPECT_FALSE(fn->empty());
}
