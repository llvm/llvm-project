//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for IntType::getABIAlignment and
// CIRDataLayout::getTypeLanguageAlignment.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/MathExtras.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;
using namespace cir;

namespace {

class IntTypeABIAlignTest : public ::testing::Test {
protected:
  IntTypeABIAlignTest() {
    context.loadDialect<cir::CIRDialect, mlir::DLTIDialect>();
  }

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

  struct Alignments {
    uint64_t abi;
    uint64_t language;
  };

  /// The ABI and language alignments of \p ty in a module whose data layout
  /// aligns i8 through i128 to their widths, as x86-64 does, and caps the
  /// language alignment of a _BitInt at \p bitIntMaxAlignBits, if given.
  /// \p recordLayouts become the module's cir.record_layouts entries.
  Alignments
  targetAlignments(mlir::Type ty,
                   std::optional<uint32_t> bitIntMaxAlignBits = {},
                   llvm::ArrayRef<mlir::NamedAttribute> recordLayouts = {}) {
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    auto i64 = mlir::IntegerType::get(&context, 64);
    llvm::SmallVector<mlir::DataLayoutEntryInterface> entries;
    for (uint64_t width : {8, 16, 32, 64, 128}) {
      auto align = mlir::DenseIntElementsAttr::get<uint64_t>(
          mlir::VectorType::get({2}, i64), {width, width});
      entries.push_back(mlir::DataLayoutEntryAttr::get(
          mlir::IntegerType::get(&context, width), align));
    }
    if (bitIntMaxAlignBits) {
      auto key = IntType::getBitIntMaxAlignKey(&context);
      auto spec = BitIntSpecAttr::get(&context, *bitIntMaxAlignBits);
      entries.push_back(mlir::DataLayoutEntryAttr::get(key, spec));
    }
    module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                    mlir::DataLayoutSpecAttr::get(&context, entries));
    if (!recordLayouts.empty())
      module->setAttr(CIRDialect::getRecordLayoutsAttrName(),
                      mlir::DictionaryAttr::get(&context, recordLayouts));
    cir::CIRDataLayout dl(module);
    Alignments result{dl.layout.getTypeABIAlignment(ty),
                      dl.getTypeLanguageAlignment(ty)};
    module->erase();
    return result;
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

// An ordinary integer's language alignment is its ABI alignment.
TEST_F(IntTypeABIAlignTest, LanguageAlignmentOfOrdinaryInt) {
  Alignments i32 = targetAlignments(IntType::get(&context, 32, true));
  EXPECT_EQ(i32.abi, 4u);
  EXPECT_EQ(i32.language, 4u);
  Alignments i128 = targetAlignments(IntType::get(&context, 128, true));
  EXPECT_EQ(i128.abi, 16u);
  EXPECT_EQ(i128.language, 16u);
}

// A _BitInt's ABI alignment is the alignment of its storage integer.
// _BitInt(17) is held in an i32.
TEST_F(IntTypeABIAlignTest, AlignmentsOfBitInt) {
  Alignments bitInt17 =
      targetAlignments(IntType::get(&context, 17, true, /*isBitInt=*/true));
  EXPECT_EQ(bitInt17.abi, 4u);
  EXPECT_EQ(bitInt17.language, 4u);
}

// x86-64 caps the language alignment of a _BitInt at 64 bits. _BitInt(65) is
// then held in an i128, whose alignment of 16 is the ABI alignment.
TEST_F(IntTypeABIAlignTest, BitIntMaxAlignCapsLanguageAlignment) {
  Alignments bitInt65 =
      targetAlignments(IntType::get(&context, 65, true, /*isBitInt=*/true), 64);
  EXPECT_EQ(bitInt65.abi, 16u);
  EXPECT_EQ(bitInt65.language, 8u);
}

// _BitInt(129) is 24 bytes, which an i192 cannot hold without padding, so it
// is held as bytes and its ABI alignment is 1.
TEST_F(IntTypeABIAlignTest, BitIntHeldAsBytes) {
  Alignments bitInt129 = targetAlignments(
      IntType::get(&context, 129, true, /*isBitInt=*/true), 64);
  EXPECT_EQ(bitInt129.abi, 1u);
  EXPECT_EQ(bitInt129.language, 8u);
}

// The int member has the higher ABI alignment, so it is the union's storage
// type and sets the union's ABI alignment. The union's language alignment is
// the largest language alignment of any member, which is the _BitInt's 8.
TEST_F(IntTypeABIAlignTest, UnionLanguageAlignmentUsesAllMembers) {
  mlir::Type members[] = {IntType::get(&context, 129, true, /*isBitInt=*/true),
                          IntType::get(&context, 32, true)};
  auto padding = cir::ArrayType::get(IntType::get(&context, 8, false), 20);
  auto unionTy = UnionType::get(&context, members, /*packed=*/false, padding,
                                RecordType::getAllDataKinds(members));
  Alignments alignments = targetAlignments(unionTy, 64);
  EXPECT_EQ(alignments.abi, 4u);
  EXPECT_EQ(alignments.language, 8u);
}

// struct Outer3 { char c; struct Inner3 { _BitInt(128) bi; } inner; } is
// packed in CIR only so that inner sits at its C offset of 8, since Inner3's
// ABI alignment is 16. Its language alignment is the record alignment in its
// layout entry. Without an entry, the packed record gets 1.
TEST_F(IntTypeABIAlignTest, PackedRecordLanguageAlignmentFromLayout) {
  auto u8 = IntType::get(&context, 8, false);
  auto s8 = IntType::get(&context, 8, true);
  mlir::Type innerMembers[] = {
      IntType::get(&context, 128, true, /*isBitInt=*/true)};
  auto inner = StructType::get(&context, innerMembers,
                               mlir::StringAttr::get(&context, "Inner3"),
                               /*packed=*/false, /*is_class=*/false,
                               RecordType::getAllDataKinds(innerMembers));
  mlir::Type outerMembers[] = {s8, cir::ArrayType::get(u8, 7), inner};
  RecordMemberKind outerKinds[] = {
      RecordMemberKind::Data, RecordMemberKind::Pad, RecordMemberKind::Data};
  auto outerName = mlir::StringAttr::get(&context, "Outer3");
  auto outer = StructType::get(&context, outerMembers, outerName,
                               /*packed=*/true, /*is_class=*/false, outerKinds);

  mlir::NamedAttribute layouts[] = {mlir::NamedAttribute(
      outerName, RecordLayoutAttr::get(&context, ArgPassingKind::CanPassInRegs,
                                       /*has_trivial_dtor=*/true,
                                       /*record_align=*/8))};
  Alignments withLayout = targetAlignments(outer, 64, layouts);
  EXPECT_EQ(withLayout.abi, 1u);
  EXPECT_EQ(withLayout.language, 8u);

  Alignments withoutLayout = targetAlignments(outer, 64);
  EXPECT_EQ(withoutLayout.language, 1u);
}

} // namespace
