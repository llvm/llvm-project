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

#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Value.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/OpenACC/CIROpenACCTypeInterfaces.h"
#include "clang/CIR/Dialect/OpenACC/RegisterOpenACCExtensions.h"
#include "gtest/gtest.h"

using namespace aiir;
using namespace cir;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class CIROpenACCPointerLikeTest : public ::testing::Test {
protected:
  CIROpenACCPointerLikeTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<cir::CIRDialect>();
    context.loadDialect<aiir::acc::OpenACCDialect>();

    // Register extension to integrate CIR types with OpenACC.
    aiir::DialectRegistry registry;
    cir::acc::registerOpenACCExtensions(registry);
    context.appendDialectRegistry(registry);
  }

  AIIRContext context;
  OpBuilder b;
  Location loc;
  llvm::StringMap<unsigned> recordNames;

  aiir::IntegerAttr getAlignOne(aiir::AIIRContext *ctx) {
    // Note that aiir::IntegerType is used instead of cir::IntType here because
    // we don't need sign information for this to be useful, so keep it simple.
    clang::CharUnits align = clang::CharUnits::One();
    return b.getI64IntegerAttr(align.getQuantity());
  }

  aiir::StringAttr getUniqueRecordName(const std::string &baseName) {
    auto it = recordNames.find(baseName);
    if (it == recordNames.end()) {
      recordNames[baseName] = 0;
      return b.getStringAttr(baseName);
    }

    return b.getStringAttr(baseName + "." +
                           std::to_string(recordNames[baseName]++));
  }

  // General handler for types without a specific test
  void testSingleType(aiir::Type ty,
                      aiir::acc::VariableTypeCategory expectedTypeCategory) {
    aiir::Type ptrTy = cir::PointerType::get(ty);

    // cir::PointerType should be castable to acc::PointerLikeType
    auto pltTy = dyn_cast_if_present<aiir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);

    EXPECT_EQ(pltTy.getElementType(), ty);

    OwningOpRef<cir::AllocaOp> varPtrOp =
        cir::AllocaOp::create(b, loc, ptrTy, ty, "", getAlignOne(&context));

    aiir::Value val = varPtrOp.get();
    aiir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<aiir::acc::PointerLikeType>>(val),
        aiir::acc::getVarType(varPtrOp.get()));

    EXPECT_EQ(typeCategory, expectedTypeCategory);
  }

  void testScalarType(aiir::Type ty) {
    testSingleType(ty, aiir::acc::VariableTypeCategory::scalar);
  }

  void testNonScalarType(aiir::Type ty) {
    testSingleType(ty, aiir::acc::VariableTypeCategory::nonscalar);
  }

  void testUncategorizedType(aiir::Type ty) {
    testSingleType(ty, aiir::acc::VariableTypeCategory::uncategorized);
  }

  void testArrayType(aiir::Type ty) {
    // Build the array pointer type.
    aiir::Type arrTy = cir::ArrayType::get(ty, 10);
    aiir::Type ptrTy = cir::PointerType::get(arrTy);

    // Verify that the pointer points to the array type..
    auto pltTy = dyn_cast_if_present<aiir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);
    EXPECT_EQ(pltTy.getElementType(), arrTy);

    // Create an alloca for the array
    OwningOpRef<cir::AllocaOp> varPtrOp =
        cir::AllocaOp::create(b, loc, ptrTy, arrTy, "", getAlignOne(&context));

    // Verify that the type category is array.
    aiir::Value val = varPtrOp.get();
    aiir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<aiir::acc::PointerLikeType>>(val),
        aiir::acc::getVarType(varPtrOp.get()));
    EXPECT_EQ(typeCategory, aiir::acc::VariableTypeCategory::array);

    // Create an array-to-pointer decay cast.
    aiir::Type ptrToElemTy = cir::PointerType::get(ty);
    OwningOpRef<cir::CastOp> decayPtr = cir::CastOp::create(
        b, loc, ptrToElemTy, cir::CastKind::array_to_ptrdecay, val);
    aiir::Value decayVal = decayPtr.get();

    // Verify that we still get the expected element type.
    auto decayPltTy =
        dyn_cast_if_present<aiir::acc::PointerLikeType>(decayVal.getType());
    ASSERT_NE(decayPltTy, nullptr);
    EXPECT_EQ(decayPltTy.getElementType(), ty);

    // Verify that we still identify the type category as an array.
    aiir::acc::VariableTypeCategory decayTypeCategory =
        decayPltTy.getPointeeTypeCategory(
            cast<TypedValue<aiir::acc::PointerLikeType>>(decayVal),
            aiir::acc::getVarType(decayPtr.get()));
    EXPECT_EQ(decayTypeCategory, aiir::acc::VariableTypeCategory::array);

    // Create an element access.
    aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
    aiir::Value index =
        cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32Ty, 2));
    OwningOpRef<cir::PtrStrideOp> accessPtr =
        cir::PtrStrideOp::create(b, loc, ptrToElemTy, decayVal, index);
    aiir::Value accessVal = accessPtr.get();

    // Verify that we still get the expected element type.
    auto accessPltTy =
        dyn_cast_if_present<aiir::acc::PointerLikeType>(accessVal.getType());
    ASSERT_NE(accessPltTy, nullptr);
    EXPECT_EQ(accessPltTy.getElementType(), ty);

    // Verify that we still identify the type category as an array.
    aiir::acc::VariableTypeCategory accessTypeCategory =
        accessPltTy.getPointeeTypeCategory(
            cast<TypedValue<aiir::acc::PointerLikeType>>(accessVal),
            aiir::acc::getVarType(accessPtr.get()));
    EXPECT_EQ(accessTypeCategory, aiir::acc::VariableTypeCategory::array);
  }

  // Structures and unions are accessed in the same way, so use a common test.
  void testRecordType(aiir::Type ty1, aiir::Type ty2,
                      cir::RecordType::RecordKind kind) {
    // Build the structure pointer type.
    cir::RecordType structTy =
        cir::RecordType::get(&context, getUniqueRecordName("S"), kind);
    structTy.complete({ty1, ty2}, false, false);
    aiir::Type ptrTy = cir::PointerType::get(structTy);

    // Verify that the pointer points to the structure type.
    auto pltTy = dyn_cast_if_present<aiir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);
    EXPECT_EQ(pltTy.getElementType(), structTy);

    // Create an alloca for the array
    OwningOpRef<cir::AllocaOp> varPtrOp = cir::AllocaOp::create(
        b, loc, ptrTy, structTy, "", getAlignOne(&context));

    // Verify that the type category is composite.
    aiir::Value val = varPtrOp.get();
    aiir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<aiir::acc::PointerLikeType>>(val),
        aiir::acc::getVarType(varPtrOp.get()));
    EXPECT_EQ(typeCategory, aiir::acc::VariableTypeCategory::composite);

    // Access the first element of the structure.
    OwningOpRef<cir::GetMemberOp> access1 = cir::GetMemberOp::create(
        b, loc, cir::PointerType::get(ty1), val, "f1", 0u);
    aiir::Value accessVal1 = access1.get();

    // Verify that we get the expected element type.
    auto access1PltTy =
        dyn_cast_if_present<aiir::acc::PointerLikeType>(accessVal1.getType());
    ASSERT_NE(access1PltTy, nullptr);
    EXPECT_EQ(access1PltTy.getElementType(), ty1);

    // Verify that the type category is still composite.
    aiir::acc::VariableTypeCategory access1TypeCategory =
        access1PltTy.getPointeeTypeCategory(
            cast<TypedValue<aiir::acc::PointerLikeType>>(accessVal1),
            aiir::acc::getVarType(access1.get()));
    EXPECT_EQ(access1TypeCategory, aiir::acc::VariableTypeCategory::composite);

    // Access the second element of the structure.
    OwningOpRef<cir::GetMemberOp> access2 = cir::GetMemberOp::create(
        b, loc, cir::PointerType::get(ty2), val, "f2", 1u);
    aiir::Value accessVal2 = access2.get();

    // Verify that we get the expected element type.
    auto access2PltTy =
        dyn_cast_if_present<aiir::acc::PointerLikeType>(accessVal2.getType());
    ASSERT_NE(access2PltTy, nullptr);
    EXPECT_EQ(access2PltTy.getElementType(), ty2);

    // Verify that the type category is still composite.
    aiir::acc::VariableTypeCategory access2TypeCategory =
        access2PltTy.getPointeeTypeCategory(
            cast<TypedValue<aiir::acc::PointerLikeType>>(accessVal2),
            aiir::acc::getVarType(access2.get()));
    EXPECT_EQ(access2TypeCategory, aiir::acc::VariableTypeCategory::composite);
  }

  void testStructType(aiir::Type ty1, aiir::Type ty2) {
    testRecordType(ty1, ty2, cir::RecordType::RecordKind::Struct);
  }

  void testUnionType(aiir::Type ty1, aiir::Type ty2) {
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
      aiir::Type ty, aiir::acc::VariableTypeCategory expectedTypeCategory) {
    // Construct a struct type with two members that are pointers to the input
    // type.
    aiir::Type ptrTy = cir::PointerType::get(ty);
    cir::RecordType structTy =
        cir::RecordType::get(&context, getUniqueRecordName("S"),
                             cir::RecordType::RecordKind::Struct);
    structTy.complete({ptrTy, ptrTy}, false, false);
    aiir::Type structPptrTy = cir::PointerType::get(structTy);

    // Create an alloca for the struct.
    OwningOpRef<cir::AllocaOp> varPtrOp = cir::AllocaOp::create(
        b, loc, structPptrTy, structTy, "S", getAlignOne(&context));
    aiir::Value val = varPtrOp.get();

    // Get a pointer to the second member.
    OwningOpRef<cir::GetMemberOp> access = cir::GetMemberOp::create(
        b, loc, cir::PointerType::get(ptrTy), val, b.getStringAttr("f2"), 1);
    aiir::Value accessVal = access.get();

    // Load the value of the second member. This is the pointer we want to test.
    OwningOpRef<cir::LoadOp> loadOp = cir::LoadOp::create(b, loc, accessVal);
    aiir::Value loadVal = loadOp.get();

    // Verify that the type category is the expected type category.
    auto pltTy = dyn_cast_if_present<aiir::acc::PointerLikeType>(ptrTy);
    aiir::acc::VariableTypeCategory typeCategory = pltTy.getPointeeTypeCategory(
        cast<TypedValue<aiir::acc::PointerLikeType>>(loadVal),
        aiir::acc::getVarType(loadOp.get()));

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
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  aiir::Type ptrTy = cir::PointerType::get(i32Ty);
  testScalarType(ptrTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToArray) {
  // Test an array type.
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testArrayType(i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToStruct) {
  // Test a struct type.
  aiir::Type i16Ty = cir::IntType::get(&context, 16, true);
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testStructType(i16Ty, i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToUnion) {
  // Test a union type.
  aiir::Type i16Ty = cir::IntType::get(&context, 16, true);
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testUnionType(i16Ty, i32Ty);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToFunction) {
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  aiir::Type funcTy =
      cir::FuncType::get(SmallVector<aiir::Type, 2>{i32Ty, i32Ty}, i32Ty);
  testNonScalarType(funcTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToVector) {
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  aiir::Type vecTy = cir::VectorType::get(i32Ty, 4);
  testNonScalarType(vecTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToVoid) {
  aiir::Type voidTy = cir::VoidType::get(&context);
  testUncategorizedType(voidTy);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToIntMember) {
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  testPointerToMemberType(i32Ty, aiir::acc::VariableTypeCategory::scalar);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToArrayMember) {
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  aiir::Type arrTy = cir::ArrayType::get(i32Ty, 10);
  testPointerToMemberType(arrTy, aiir::acc::VariableTypeCategory::array);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerToStructMember) {
  aiir::Type i32Ty = cir::IntType::get(&context, 32, true);
  cir::RecordType structTy = cir::RecordType::get(
      &context, getUniqueRecordName("S"), cir::RecordType::RecordKind::Struct);
  structTy.complete({i32Ty, i32Ty}, false, false);
  testPointerToMemberType(structTy, aiir::acc::VariableTypeCategory::composite);
}
