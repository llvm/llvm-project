//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CIR implementation of OpenACC's PointertLikeType interface
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/OpenACC/CIROpenACCTypeInterfaces.h"
#include "clang/CIR/Dialect/OpenACC/RegisterOpenACCExtensions.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class CIROpenACCPointerLikeTest : public ::testing::Test {
protected:
  CIROpenACCPointerLikeTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<cir::CIRDialect>();
    context.loadDialect<mlir::acc::OpenACCDialect>();

    // Register extension to integrate CIR types with OpenACC.
    mlir::DialectRegistry registry;
    cir::acc::registerOpenACCExtensions(registry);
    context.appendDialectRegistry(registry);
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;

  mlir::IntegerAttr getSizeFromCharUnits(mlir::MLIRContext *ctx,
                                         clang::CharUnits size) {
    // Note that mlir::IntegerType is used instead of cir::IntType here
    // because we don't need sign information for this to be useful, so keep
    // it simple.
    return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                  size.getQuantity());
  }

  // General handler for types without a specific test
  void testElementType(mlir::Type ty) {
    mlir::Type ptrTy = cir::PointerType::get(ty);

    // cir::PointerType should be castable to acc::PointerLikeType
    auto pltTy = dyn_cast_if_present<mlir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);

    EXPECT_EQ(pltTy.getElementType(), ty);

    OwningOpRef<cir::AllocaOp> varPtrOp = b.create<cir::AllocaOp>(
        loc, ptrTy, ty, "",
        getSizeFromCharUnits(&context, clang::CharUnits::One()));

    mlir::Value val = varPtrOp.get();
    mlir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<mlir::acc::PointerLikeType>>(val),
        mlir::acc::getVarType(varPtrOp.get()));

    if (isAnyIntegerOrFloatingPointType(ty) ||
        mlir::isa<cir::PointerType>(ty) || mlir::isa<cir::BoolType>(ty)) {
      EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::scalar);
    } else if (mlir::isa<cir::ArrayType>(ty)) {
      EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::array);
    } else if (mlir::isa<cir::RecordType>(ty)) {
      EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::composite);
    } else if (mlir::isa<cir::FuncType, cir::VectorType>(ty)) {
      EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::nonscalar);
    } else if (mlir::isa<cir::VoidType>(ty)) {
      EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::uncategorized);
    } else {
      EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::uncategorized);
      // If we hit this, we need to add support for a new type.
      ASSERT_TRUE(false);
    }
  }
};

TEST_F(CIROpenACCPointerLikeTest, testPointerToInt) {
  // Test various scalar types.
  testElementType(cir::IntType::get(&context, 8, true));
  testElementType(cir::IntType::get(&context, 8, false));
  testElementType(cir::IntType::get(&context, 16, true));
  testElementType(cir::IntType::get(&context, 16, false));
  testElementType(cir::IntType::get(&context, 32, true));
  testElementType(cir::IntType::get(&context, 32, false));
  testElementType(cir::IntType::get(&context, 64, true));
  testElementType(cir::IntType::get(&context, 64, false));
  testElementType(cir::IntType::get(&context, 128, true));
  testElementType(cir::IntType::get(&context, 128, false));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToBool) {
  testElementType(cir::BoolType::get(&context));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToFloat) {
  testElementType(cir::SingleType::get(&context));
  testElementType(cir::DoubleType::get(&context));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToPointer) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  mlir::Type ptrTy = cir::PointerType::get(i32Ty);
  testElementType(ptrTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToArray) {
  // Test an array type.
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testElementType(cir::ArrayType::get(i32Ty, 10));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToStruct) {
  // Test a struct type.
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  llvm::ArrayRef<mlir::Type> fields = {i32Ty, i32Ty};
  cir::RecordType structTy = cir::RecordType::get(
      &context, b.getStringAttr("S"), cir::RecordType::RecordKind::Struct);
  structTy.complete(fields, false, false);
  testElementType(structTy);

  // Test a union type.
  cir::RecordType unionTy = cir::RecordType::get(
      &context, b.getStringAttr("U"), cir::RecordType::RecordKind::Union);
  unionTy.complete(fields, false, false);
  testElementType(unionTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToFunction) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  cir::FuncType::get(SmallVector<mlir::Type, 2>{i32Ty, i32Ty}, i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToVector) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  mlir::Type vecTy = cir::VectorType::get(i32Ty, 4);
  testElementType(vecTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToVoid) {
  mlir::Type voidTy = cir::VoidType::get(&context);
  testElementType(voidTy);
}
