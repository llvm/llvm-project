//===- FIROpenACCPointerLikeTypeInterfaceTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/OpenACC/Support/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/Support/InitFIR.h"

using namespace mlir;

namespace {

struct FIROpenACCPointerLikeTypeInterfaceTest : public testing::Test {
  void SetUp() override {
    mlir::DialectRegistry registry;
    fir::acc::registerOpenACCExtensions(registry);
    context.appendDialectRegistry(registry);
    fir::support::loadDialects(context);
    kindMap = std::make_unique<fir::KindMapping>(&context);
    module = ModuleOp::create(UnknownLoc::get(&context));
    fir::setKindMapping(module, *kindMap);
  }

  MLIRContext context;
  std::unique_ptr<fir::KindMapping> kindMap;
  ModuleOp module;
};

TEST_F(FIROpenACCPointerLikeTypeInterfaceTest,
    GetAsMemRefTypeFromFirRefStaticArray) {
  Type f32 = Float32Type::get(&context);
  Type seq = fir::SequenceType::get({10}, f32);
  Type refTy = fir::ReferenceType::get(seq);
  auto ptrLike = cast<acc::PointerLikeType>(refTy);

  MemRefType memrefTy = ptrLike.getAsMemRefType(module);
  ASSERT_TRUE(memrefTy);
  EXPECT_EQ(memrefTy.getRank(), 1);
  EXPECT_EQ(memrefTy.getDimSize(0), 10);
  EXPECT_TRUE(isa<Float32Type>(memrefTy.getElementType()));
}

TEST_F(FIROpenACCPointerLikeTypeInterfaceTest,
    GetAsMemRefTypeFromFirRefMultiDimArray) {
  Type i32 = IntegerType::get(&context, 32);
  Type seq = fir::SequenceType::get({2, 3}, i32);
  Type refTy = fir::ReferenceType::get(seq);
  auto ptrLike = cast<acc::PointerLikeType>(refTy);

  MemRefType memrefTy = ptrLike.getAsMemRefType(module);
  ASSERT_TRUE(memrefTy);
  EXPECT_EQ(memrefTy.getRank(), 2);
  // MemRef layout is in reversed order:
  EXPECT_EQ(memrefTy.getDimSize(0), 3);
  EXPECT_EQ(memrefTy.getDimSize(1), 2);
  EXPECT_TRUE(isa<IntegerType>(memrefTy.getElementType()));
  EXPECT_EQ(cast<IntegerType>(memrefTy.getElementType()).getWidth(), 32u);
}

TEST_F(FIROpenACCPointerLikeTypeInterfaceTest,
    GetAsMemRefTypeFromFirHeapDynamicArray) {
  Type f64 = Float64Type::get(&context);
  Type seq = fir::SequenceType::get({ShapedType::kDynamic}, f64);
  Type heapTy = fir::HeapType::get(seq);
  auto ptrLike = cast<acc::PointerLikeType>(heapTy);

  MemRefType memrefTy = ptrLike.getAsMemRefType(module);
  ASSERT_TRUE(memrefTy);
  EXPECT_EQ(memrefTy.getRank(), 1);
  EXPECT_EQ(memrefTy.getDimSize(0), ShapedType::kDynamic);
  EXPECT_TRUE(isa<Float64Type>(memrefTy.getElementType()));
}

TEST_F(FIROpenACCPointerLikeTypeInterfaceTest,
    GetAsMemRefTypeFromFirPointerStaticArray) {
  Type f32 = Float32Type::get(&context);
  Type seq = fir::SequenceType::get({4}, f32);
  Type ptrTy = fir::PointerType::get(seq);
  auto ptrLike = cast<acc::PointerLikeType>(ptrTy);

  MemRefType memrefTy = ptrLike.getAsMemRefType(module);
  ASSERT_TRUE(memrefTy);
  EXPECT_EQ(memrefTy.getRank(), 1);
  EXPECT_EQ(memrefTy.getDimSize(0), 4);
}

TEST_F(FIROpenACCPointerLikeTypeInterfaceTest,
    GetAsMemRefTypeFromFirRefScalarInteger) {
  Type i32 = IntegerType::get(&context, 32);
  Type refTy = fir::ReferenceType::get(i32);
  auto ptrLike = cast<acc::PointerLikeType>(refTy);

  MemRefType memrefTy = ptrLike.getAsMemRefType(module);
  ASSERT_TRUE(memrefTy);
  EXPECT_EQ(memrefTy.getRank(), 0);
  EXPECT_TRUE(isa<IntegerType>(memrefTy.getElementType()));
  EXPECT_EQ(cast<IntegerType>(memrefTy.getElementType()).getWidth(), 32u);
}

TEST_F(FIROpenACCPointerLikeTypeInterfaceTest,
    GetAsMemRefTypeFromFirRefScalarLogical) {
  constexpr unsigned logicalKind = 4;
  Type logicalTy = fir::LogicalType::get(&context, logicalKind);
  Type refTy = fir::ReferenceType::get(logicalTy);
  auto ptrLike = cast<acc::PointerLikeType>(refTy);

  MemRefType memrefTy = ptrLike.getAsMemRefType(module);
  ASSERT_TRUE(memrefTy);
  EXPECT_EQ(memrefTy.getRank(), 0);
  auto elTy = dyn_cast<IntegerType>(memrefTy.getElementType());
  ASSERT_TRUE(elTy);
  EXPECT_EQ(elTy.getWidth(), kindMap->getLogicalBitsize(logicalKind));
}

} // namespace
