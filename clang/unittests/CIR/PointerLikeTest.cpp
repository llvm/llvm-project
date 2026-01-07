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
  llvm::StringMap<unsigned> recordNames;

  mlir::IntegerAttr getAlignOne(mlir::MLIRContext *ctx) {
    // Note that mlir::IntegerType is used instead of cir::IntType here because
    // we don't need sign information for this to be useful, so keep it simple.
    clang::CharUnits align = clang::CharUnits::One();
    return b.getI64IntegerAttr(align.getQuantity());
  }

  mlir::StringAttr getUniqueRecordName(const std::string &baseName) {
    auto it = recordNames.find(baseName);
    if (it == recordNames.end()) {
      recordNames[baseName] = 0;
      return b.getStringAttr(baseName);
    }

    return b.getStringAttr(baseName + "." +
                           std::to_string(recordNames[baseName]++));
  }

  // General handler for types without a specific test
  void testSingleType(mlir::Type ty,
                      mlir::acc::VariableTypeCategory expectedTypeCategory) {
    mlir::Type ptrTy = cir::PointerType::get(ty);

    // cir::PointerType should be castable to acc::PointerLikeType
    auto pltTy = dyn_cast_if_present<mlir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);

    EXPECT_EQ(pltTy.getElementType(), ty);

    OwningOpRef<cir::AllocaOp> varPtrOp =
        cir::AllocaOp::create(b, loc, ptrTy, ty, "", getAlignOne(&context));

    mlir::Value val = varPtrOp.get();
    mlir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<mlir::acc::PointerLikeType>>(val),
        mlir::acc::getVarType(varPtrOp.get()));

    EXPECT_EQ(typeCategory, expectedTypeCategory);
  }

  void testScalarType(mlir::Type ty) {
    testSingleType(ty, mlir::acc::VariableTypeCategory::scalar);
  }

  void testNonScalarType(mlir::Type ty) {
    testSingleType(ty, mlir::acc::VariableTypeCategory::nonscalar);
  }

  void testUncategorizedType(mlir::Type ty) {
    testSingleType(ty, mlir::acc::VariableTypeCategory::uncategorized);
  }

  void testArrayType(mlir::Type ty) {
    // Build the array pointer type.
    mlir::Type arrTy = cir::ArrayType::get(ty, 10);
    mlir::Type ptrTy = cir::PointerType::get(arrTy);

    // Verify that the pointer points to the array type..
    auto pltTy = dyn_cast_if_present<mlir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);
    EXPECT_EQ(pltTy.getElementType(), arrTy);

    // Create an alloca for the array
    OwningOpRef<cir::AllocaOp> varPtrOp =
        cir::AllocaOp::create(b, loc, ptrTy, arrTy, "", getAlignOne(&context));

    // Verify that the type category is array.
    mlir::Value val = varPtrOp.get();
    mlir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<mlir::acc::PointerLikeType>>(val),
        mlir::acc::getVarType(varPtrOp.get()));
    EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::array);

    // Create an array-to-pointer decay cast.
    mlir::Type ptrToElemTy = cir::PointerType::get(ty);
    OwningOpRef<cir::CastOp> decayPtr = cir::CastOp::create(
        b, loc, ptrToElemTy, cir::CastKind::array_to_ptrdecay, val);
    mlir::Value decayVal = decayPtr.get();

    // Verify that we still get the expected element type.
    auto decayPltTy =
        dyn_cast_if_present<mlir::acc::PointerLikeType>(decayVal.getType());
    ASSERT_NE(decayPltTy, nullptr);
    EXPECT_EQ(decayPltTy.getElementType(), ty);

    // Verify that we still identify the type category as an array.
    mlir::acc::VariableTypeCategory decayTypeCategory =
        decayPltTy.getPointeeTypeCategory(
            cast<TypedValue<mlir::acc::PointerLikeType>>(decayVal),
            mlir::acc::getVarType(decayPtr.get()));
    EXPECT_EQ(decayTypeCategory, mlir::acc::VariableTypeCategory::array);

    // Create an element access.
    mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
    mlir::Value index =
        cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32Ty, 2));
    OwningOpRef<cir::PtrStrideOp> accessPtr =
        cir::PtrStrideOp::create(b, loc, ptrToElemTy, decayVal, index);
    mlir::Value accessVal = accessPtr.get();

    // Verify that we still get the expected element type.
    auto accessPltTy =
        dyn_cast_if_present<mlir::acc::PointerLikeType>(accessVal.getType());
    ASSERT_NE(accessPltTy, nullptr);
    EXPECT_EQ(accessPltTy.getElementType(), ty);

    // Verify that we still identify the type category as an array.
    mlir::acc::VariableTypeCategory accessTypeCategory =
        accessPltTy.getPointeeTypeCategory(
            cast<TypedValue<mlir::acc::PointerLikeType>>(accessVal),
            mlir::acc::getVarType(accessPtr.get()));
    EXPECT_EQ(accessTypeCategory, mlir::acc::VariableTypeCategory::array);
  }

  // Structures and unions are accessed in the same way, so use a common test.
  void testRecordType(mlir::Type ty1, mlir::Type ty2,
                      cir::RecordType::RecordKind kind) {
    // Build the structure pointer type.
    cir::RecordType structTy =
        cir::RecordType::get(&context, getUniqueRecordName("S"), kind);
    structTy.complete({ty1, ty2}, false, false);
    mlir::Type ptrTy = cir::PointerType::get(structTy);

    // Verify that the pointer points to the structure type.
    auto pltTy = dyn_cast_if_present<mlir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);
    EXPECT_EQ(pltTy.getElementType(), structTy);

    // Create an alloca for the array
    OwningOpRef<cir::AllocaOp> varPtrOp = cir::AllocaOp::create(
        b, loc, ptrTy, structTy, "", getAlignOne(&context));

    // Verify that the type category is composite.
    mlir::Value val = varPtrOp.get();
    mlir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<mlir::acc::PointerLikeType>>(val),
        mlir::acc::getVarType(varPtrOp.get()));
    EXPECT_EQ(typeCategory, mlir::acc::VariableTypeCategory::composite);

    // Access the first element of the structure.
    OwningOpRef<cir::GetMemberOp> access1 = cir::GetMemberOp::create(
        b, loc, cir::PointerType::get(ty1), val, "f1", 0u);
    mlir::Value accessVal1 = access1.get();

    // Verify that we get the expected element type.
    auto access1PltTy =
        dyn_cast_if_present<mlir::acc::PointerLikeType>(accessVal1.getType());
    ASSERT_NE(access1PltTy, nullptr);
    EXPECT_EQ(access1PltTy.getElementType(), ty1);

    // Verify that the type category is still composite.
    mlir::acc::VariableTypeCategory access1TypeCategory =
        access1PltTy.getPointeeTypeCategory(
            cast<TypedValue<mlir::acc::PointerLikeType>>(accessVal1),
            mlir::acc::getVarType(access1.get()));
    EXPECT_EQ(access1TypeCategory, mlir::acc::VariableTypeCategory::composite);

    // Access the second element of the structure.
    OwningOpRef<cir::GetMemberOp> access2 = cir::GetMemberOp::create(
        b, loc, cir::PointerType::get(ty2), val, "f2", 1u);
    mlir::Value accessVal2 = access2.get();

    // Verify that we get the expected element type.
    auto access2PltTy =
        dyn_cast_if_present<mlir::acc::PointerLikeType>(accessVal2.getType());
    ASSERT_NE(access2PltTy, nullptr);
    EXPECT_EQ(access2PltTy.getElementType(), ty2);

    // Verify that the type category is still composite.
    mlir::acc::VariableTypeCategory access2TypeCategory =
        access2PltTy.getPointeeTypeCategory(
            cast<TypedValue<mlir::acc::PointerLikeType>>(accessVal2),
            mlir::acc::getVarType(access2.get()));
    EXPECT_EQ(access2TypeCategory, mlir::acc::VariableTypeCategory::composite);
  }

  void testStructType(mlir::Type ty1, mlir::Type ty2) {
    testRecordType(ty1, ty2, cir::RecordType::RecordKind::Struct);
  }

  void testUnionType(mlir::Type ty1, mlir::Type ty2) {
    testRecordType(ty1, ty2, cir::RecordType::RecordKind::Union);
  }

  // This is testing a case like this:
  //
  // struct S {
  //   int *f1;
  //   int *f2;
  // } *p;
  // int *pMember = p->f2;
  //
  // That is, we are not testing a pointer to a member, we're testing a pointer
  // that is loaded as a member value.
  void testPointerToMemberType(
      mlir::Type ty, mlir::acc::VariableTypeCategory expectedTypeCategory) {
    // Construct a struct type with two members that are pointers to the input
    // type.
    mlir::Type ptrTy = cir::PointerType::get(ty);
    cir::RecordType structTy =
        cir::RecordType::get(&context, getUniqueRecordName("S"),
                             cir::RecordType::RecordKind::Struct);
    structTy.complete({ptrTy, ptrTy}, false, false);
    mlir::Type structPptrTy = cir::PointerType::get(structTy);

    // Create an alloca for the struct.
    OwningOpRef<cir::AllocaOp> varPtrOp = cir::AllocaOp::create(
        b, loc, structPptrTy, structTy, "S", getAlignOne(&context));
    mlir::Value val = varPtrOp.get();

    // Get a pointer to the second member.
    OwningOpRef<cir::GetMemberOp> access = cir::GetMemberOp::create(
        b, loc, cir::PointerType::get(ptrTy), val, b.getStringAttr("f2"), 1);
    mlir::Value accessVal = access.get();

    // Load the value of the second member. This is the pointer we want to test.
    OwningOpRef<cir::LoadOp> loadOp = cir::LoadOp::create(b, loc, accessVal);
    mlir::Value loadVal = loadOp.get();

    // Verify that the type category is the expected type category.
    auto pltTy = dyn_cast_if_present<mlir::acc::PointerLikeType>(ptrTy);
    mlir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<mlir::acc::PointerLikeType>>(loadVal),
        mlir::acc::getVarType(loadOp.get()));

    EXPECT_EQ(typeCategory, expectedTypeCategory);
  }
};

TEST_F(CIROpenACCPointerLikeTest, testPointerToInt) {
  // Test various scalar types.
  testScalarType(cir::IntType::get(&context, 8, true));
  testScalarType(cir::IntType::get(&context, 8, false));
  testScalarType(cir::IntType::get(&context, 16, true));
  testScalarType(cir::IntType::get(&context, 16, false));
  testScalarType(cir::IntType::get(&context, 32, true));
  testScalarType(cir::IntType::get(&context, 32, false));
  testScalarType(cir::IntType::get(&context, 64, true));
  testScalarType(cir::IntType::get(&context, 64, false));
  testScalarType(cir::IntType::get(&context, 128, true));
  testScalarType(cir::IntType::get(&context, 128, false));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToBool) {
  testScalarType(cir::BoolType::get(&context));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToFloat) {
  testScalarType(cir::SingleType::get(&context));
  testScalarType(cir::DoubleType::get(&context));
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToPointer) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  mlir::Type ptrTy = cir::PointerType::get(i32Ty);
  testScalarType(ptrTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToArray) {
  // Test an array type.
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testArrayType(i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToStruct) {
  // Test a struct type.
  mlir::Type i16Ty = cir::IntType::get(&context, 16, true);
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testStructType(i16Ty, i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToUnion) {
  // Test a union type.
  mlir::Type i16Ty = cir::IntType::get(&context, 16, true);
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testUnionType(i16Ty, i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToFunction) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  mlir::Type funcTy =
      cir::FuncType::get(SmallVector<mlir::Type, 2>{i32Ty, i32Ty}, i32Ty);
  testNonScalarType(funcTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToVector) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  mlir::Type vecTy = cir::VectorType::get(i32Ty, 4);
  testNonScalarType(vecTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToVoid) {
  mlir::Type voidTy = cir::VoidType::get(&context);
  testUncategorizedType(voidTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToIntMember) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testPointerToMemberType(i32Ty, mlir::acc::VariableTypeCategory::scalar);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToArrayMember) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  mlir::Type arrTy = cir::ArrayType::get(i32Ty, 10);
  testPointerToMemberType(arrTy, mlir::acc::VariableTypeCategory::array);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToStructMember) {
  mlir::Type i32Ty = cir::IntType::get(&context, 32, true);
  cir::RecordType structTy = cir::RecordType::get(
      &context, getUniqueRecordName("S"), cir::RecordType::RecordKind::Struct);
  structTy.complete({i32Ty, i32Ty}, false, false);
  testPointerToMemberType(structTy, mlir::acc::VariableTypeCategory::composite);
}
