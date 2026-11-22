//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for VectorType::getABIAlignment and the language alignment of
// vectors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;
using namespace cir;

namespace {

class VectorTypeABIAlignTest : public ::testing::Test {
protected:
  VectorTypeABIAlignTest() {
    context.loadDialect<cir::CIRDialect, mlir::DLTIDialect>();
  }

  MLIRContext context;

  struct Alignments {
    uint64_t abi;
    uint64_t language;
  };

  /// The ABI and language alignments of a vector of \p size elements of
  /// \p elementType, in a module whose #cir.vector_spec entry caps vector
  /// alignment at \p maxAlignBits, if given.
  Alignments alignments(mlir::Type elementType, uint64_t size,
                        std::optional<uint32_t> maxAlignBits = std::nullopt) {
    cir::VectorType ty = cir::VectorType::getFixed(elementType, size);
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    if (maxAlignBits) {
      auto key = cir::VectorType::getMaxAlignKey(&context);
      auto spec = cir::VectorSpecAttr::get(&context, *maxAlignBits);
      auto entry = mlir::DataLayoutEntryAttr::get(key, spec);
      module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                      mlir::DataLayoutSpecAttr::get(&context, {entry}));
    }
    cir::CIRDataLayout dl(module);
    Alignments result{dl.layout.getTypeABIAlignment(ty),
                      dl.getTypeLanguageAlignment(ty)};
    module->erase();
    return result;
  }

  uint64_t abiAlign(mlir::Type elementType, uint64_t size) {
    return alignments(elementType, size).abi;
  }

  mlir::Type f32() { return cir::SingleType::get(&context); }
  mlir::Type f64() { return cir::DoubleType::get(&context); }
  mlir::Type u8() { return cir::IntType::get(&context, 8, /*isSigned=*/false); }
};

// The hook answers in bytes, so a vector holding one double aligns to 8 and
// not to its 64-bit size.
TEST_F(VectorTypeABIAlignTest, AlignmentIsInBytes) {
  EXPECT_EQ(abiAlign(f64(), 1), 8u);
  EXPECT_EQ(abiAlign(f32(), 1), 4u);
}

// A size that is already a power of two keeps it rather than rounding to the
// next one up.
TEST_F(VectorTypeABIAlignTest, ExactPowerOfTwoSizeIsUnchanged) {
  EXPECT_EQ(abiAlign(f32(), 4), 16u);
  EXPECT_EQ(abiAlign(f32(), 8), 32u);
  EXPECT_EQ(abiAlign(f32(), 16), 64u);
}

// An element count that is not a power of two rounds the byte size up.
TEST_F(VectorTypeABIAlignTest, NonPowerOfTwoElementCountRoundsUp) {
  EXPECT_EQ(abiAlign(f32(), 3), 16u);
  EXPECT_EQ(abiAlign(f64(), 3), 32u);
}

// AArch64's MaxVectorAlign is 128 bits. The language alignment of a vector
// wider than 16 bytes is 16. The ABI alignment is not capped.
TEST_F(VectorTypeABIAlignTest, MaxAlignCapsLanguageAlignment) {
  Alignments v8f32 = alignments(f32(), 8, 128);
  EXPECT_EQ(v8f32.abi, 32u);
  EXPECT_EQ(v8f32.language, 16u);

  Alignments v4f32 = alignments(f32(), 4, 128);
  EXPECT_EQ(v4f32.abi, 16u);
  EXPECT_EQ(v4f32.language, 16u);

  Alignments v1f32 = alignments(f32(), 1, 128);
  EXPECT_EQ(v1f32.abi, 4u);
  EXPECT_EQ(v1f32.language, 4u);

  // 17 x i8 occupies 17 bytes. Its ABI alignment is 32 bytes.
  Alignments v17u8 = alignments(u8(), 17, 128);
  EXPECT_EQ(v17u8.abi, 32u);
  EXPECT_EQ(v17u8.language, 16u);
}

// A cap above the ABI alignment leaves the language alignment equal to it.
TEST_F(VectorTypeABIAlignTest, MaxAlignAboveABIAlignIsUnused) {
  Alignments v8f32 = alignments(f32(), 8, 512);
  EXPECT_EQ(v8f32.abi, 32u);
  EXPECT_EQ(v8f32.language, 32u);
}

// Without a #cir.vector_spec entry the language alignment equals the ABI
// alignment.
TEST_F(VectorTypeABIAlignTest, NoMaxAlignEntry) {
  Alignments v8f32 = alignments(f32(), 8);
  EXPECT_EQ(v8f32.abi, 32u);
  EXPECT_EQ(v8f32.language, 32u);
}

// The cap applies to the language alignment only. A 36-byte v9f32 keeps an
// ABI alignment of 64, which sets its allocation size, the stride of an array
// of it, and its placement in a record, as for the LLVM type <9 x float>.
TEST_F(VectorTypeABIAlignTest, MaxAlignLeavesABIAlignment) {
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  auto key = cir::VectorType::getMaxAlignKey(&context);
  auto spec = cir::VectorSpecAttr::get(&context, 128);
  auto entry = mlir::DataLayoutEntryAttr::get(key, spec);
  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                  mlir::DataLayoutSpecAttr::get(&context, {entry}));
  cir::CIRDataLayout dl(module);

  cir::VectorType v9f32 = cir::VectorType::getFixed(f32(), 9);
  EXPECT_EQ(dl.layout.getTypeABIAlignment(v9f32), 64u);
  EXPECT_EQ(dl.getTypeLanguageAlignment(v9f32), 16u);
  EXPECT_EQ(dl.layout.getTypeSize(v9f32), 36u);
  EXPECT_EQ(dl.layout.getTypeAllocSize(v9f32), 64u);

  auto arrayTy = cir::ArrayType::get(v9f32, 2);
  EXPECT_EQ(dl.layout.getTypeSize(arrayTy), 128u);
  EXPECT_EQ(dl.layout.getTypeABIAlignment(arrayTy), 64u);
  EXPECT_EQ(dl.getTypeLanguageAlignment(arrayTy), 16u);

  auto structTy = cir::StructType::get(
      &context, mlir::StringAttr::get(&context, "T"), /*is_class=*/false);
  mlir::Type members[] = {v9f32,
                          cir::IntType::get(&context, 32, /*isSigned=*/true)};
  structTy.complete(members, /*packed=*/false,
                    cir::RecordType::getAllDataKinds(members));
  EXPECT_EQ(structTy.getElementOffset(dl.layout, 1), 64u);
  EXPECT_EQ(dl.layout.getTypeSize(structTy), 128u);
  EXPECT_EQ(dl.layout.getTypeABIAlignment(structTy), 64u);
  EXPECT_EQ(dl.getTypeLanguageAlignment(structTy), 16u);

  module->erase();
}

} // namespace
