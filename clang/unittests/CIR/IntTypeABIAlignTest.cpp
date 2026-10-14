//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for IntType::getABIAlignment.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/MathExtras.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

namespace {

class IntTypeABIAlignTest : public ::testing::Test {
protected:
  IntTypeABIAlignTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;

  uint64_t abiAlign(unsigned width, bool isSigned) {
    IntType ty = IntType::get(&context, width, isSigned);
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    mlir::DataLayout dl(module);
    uint64_t align = dl.getTypeABIAlignment(ty);
    module->erase();
    return align;
  }
};

// The fundamental integer widths keep their natural alignment.
TEST_F(IntTypeABIAlignTest, FundamentalWidths) {
  EXPECT_EQ(abiAlign(8, true), 1u);
  EXPECT_EQ(abiAlign(16, true), 2u);
  EXPECT_EQ(abiAlign(32, true), 4u);
  EXPECT_EQ(abiAlign(64, true), 8u);
}

// __int128 is 16-byte aligned, so a 128-bit integer must not round down.
TEST_F(IntTypeABIAlignTest, Int128) { EXPECT_EQ(abiAlign(128, true), 16u); }

// A non-fundamental width must report a power-of-two alignment, not the bare
// width / 8 (which would be 3 for i24 and trip llvm::Align).  Rounding is on
// the bit width, so a width that isn't a byte multiple still rounds up like
// LLVM's DataLayout (i17 aligns like i32).
TEST_F(IntTypeABIAlignTest, NonFundamentalWidthIsPowerOfTwo) {
  for (unsigned width : {17u, 24u}) {
    uint64_t align = abiAlign(width, true);
    EXPECT_TRUE(llvm::isPowerOf2_64(align))
        << "i" << width << " alignment " << align << " is not a power of two";
  }
  EXPECT_EQ(abiAlign(17, true), 4u);
  EXPECT_EQ(abiAlign(24, true), 4u);
}

// A sub-byte width must still report a valid, non-zero alignment.
TEST_F(IntTypeABIAlignTest, SubByteWidth) { EXPECT_EQ(abiAlign(1, true), 1u); }

} // namespace
