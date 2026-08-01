//===- FIROpenACCSupportAnalysisTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Analysis/FIROpenACCSupportAnalysis.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"

using namespace mlir;

namespace {

struct FIROpenACCSupportAnalysisTest : public testing::Test {
  void SetUp() override {
    context.loadDialect<fir::FIROpsDialect, DLTIDialect, LLVM::LLVMDialect>();
    kindMap = std::make_unique<fir::KindMapping>(&context);
    module = ModuleOp::create(UnknownLoc::get(&context));
    fir::setKindMapping(module, *kindMap);
    support.setImplementation(fir::acc::FIROpenACCSupportAnalysis());
  }

  std::optional<acc::TypeSizeAndAlignment> getExpectedFIRSize(Type ty) {
    std::optional<DataLayout> dl = acc::getDataLayout(module);
    if (!dl)
      return std::nullopt;
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
        /*forceUnifiedTBAATree=*/false, *dl);
    std::optional<std::pair<uint64_t, unsigned short>> sizeAndAlignment =
        fir::getTypeSizeAndAlignment(
            UnknownLoc::get(&context), ty, *dl, typeConverter.getKindMap());
    if (!sizeAndAlignment)
      return std::nullopt;
    return acc::TypeSizeAndAlignment{
        llvm::TypeSize::getFixed(sizeAndAlignment->first),
        llvm::TypeSize::getFixed(sizeAndAlignment->second)};
  }

  MLIRContext context;
  std::unique_ptr<fir::KindMapping> kindMap;
  ModuleOp module;
  acc::OpenACCSupport support;
};

TEST_F(FIROpenACCSupportAnalysisTest, FIRLogicalScalarSizeAndAlignment) {
  Type logicalTy = fir::LogicalType::get(&context, /*kind=*/4);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(logicalTy, module);
  std::optional<acc::TypeSizeAndAlignment> expected =
      getExpectedFIRSize(logicalTy);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, FIRCharacterScalarSizeAndAlignment) {
  Type charTy = fir::CharacterType::get(&context, /*kind=*/1, /*len=*/8);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(charTy, module);
  std::optional<acc::TypeSizeAndAlignment> expected =
      getExpectedFIRSize(charTy);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, FIRReferenceTypeUsesPointerSize) {
  Type logicalTy = fir::LogicalType::get(&context, /*kind=*/4);
  Type refTy = fir::ReferenceType::get(logicalTy);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(refTy, module);
  LLVM::LLVMPointerType ptrTy = LLVM::LLVMPointerType::get(&context);
  std::optional<acc::TypeSizeAndAlignment> expected =
      acc::getTypeSizeAndAlignment(ptrTy, module);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, BuiltinTypeDelegatesToAccUtilities) {
  Type i32 = IntegerType::get(&context, 32);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(i32, module);
  std::optional<acc::TypeSizeAndAlignment> expected =
      acc::getTypeSizeAndAlignment(i32, module);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  DataLayout dl(module);
  EXPECT_EQ(result->first, dl.getTypeSize(i32));
  EXPECT_EQ(result->second.getFixedValue(), dl.getTypeABIAlignment(i32));
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, TupleWithFIRArrayMemberSizeAndAlignment) {
  Type f32 = Float32Type::get(&context);
  Type seqTy = fir::SequenceType::get({1024}, f32);
  Type tupleTy = TupleType::get(&context, {seqTy});
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(tupleTy, module);
  std::optional<acc::TypeSizeAndAlignment> expected = getExpectedFIRSize(seqTy);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, FIRBoxTypeSizeAndAlignment) {
  Type f32 = Float32Type::get(&context);
  Type seqTy = fir::SequenceType::get({4, 3}, f32);
  Type boxTy = fir::BoxType::get(seqTy);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(boxTy, module);
  ASSERT_TRUE(result.has_value());

  std::optional<DataLayout> dl = acc::getDataLayout(module);
  ASSERT_TRUE(dl.has_value());
  fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
      /*forceUnifiedTBAATree=*/false, *dl);
  Type structTy =
      typeConverter.convertBoxTypeAsStruct(cast<fir::BaseBoxType>(boxTy));
  std::optional<acc::TypeSizeAndAlignment> expected =
      acc::getTypeSizeAndAlignment(structTy, module, *dl, &support);
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

} // namespace
