//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CIR implementation of OpenACC's PointerLikeType interface
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
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
        cir::AllocaOp::create(b, loc, ptrTy, "", getAlignOne(&context));

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
        cir::AllocaOp::create(b, loc, ptrTy, "", getAlignOne(&context));

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
    OwningOpRef<cir::ConstantOp> indexOp =
        cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32Ty, 2));
    mlir::Value index = indexOp.get();
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
  void testRecordType(mlir::Type ty1, mlir::Type ty2, bool is_union) {
    // Build the structure/union type.
    cir::RecordType structTy;
    if (is_union)
      structTy = cir::UnionType::get(&context, getUniqueRecordName("S"));
    else
      structTy = cir::StructType::get(&context, getUniqueRecordName("S"),
                                      /*is_class=*/false);
    mlir::Type members[] = {ty1, ty2};
    structTy.complete(members, /*packed=*/false, /*padding=*/{},
                      cir::RecordType::getAllDataKinds(members));
    mlir::Type ptrTy = cir::PointerType::get(structTy);

    // Verify that the pointer points to the structure type.
    auto pltTy = dyn_cast_if_present<mlir::acc::PointerLikeType>(ptrTy);
    ASSERT_NE(pltTy, nullptr);
    EXPECT_EQ(pltTy.getElementType(), structTy);

    // Create an alloca for the array
    OwningOpRef<cir::AllocaOp> varPtrOp =
        cir::AllocaOp::create(b, loc, ptrTy, "", getAlignOne(&context));

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
    testRecordType(ty1, ty2, /*is_union=*/false);
  }

  void testUnionType(mlir::Type ty1, mlir::Type ty2) {
    testRecordType(ty1, ty2, /*is_union=*/true);
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
        cir::StructType::get(&context, getUniqueRecordName("S"),
                             /*is_class=*/false);
    mlir::Type members[] = {ptrTy, ptrTy};
    structTy.complete(members, /*packed=*/false, /*padding=*/{},
                      cir::RecordType::getAllDataKinds(members));
    mlir::Type structPptrTy = cir::PointerType::get(structTy);

    // Create an alloca for the struct.
    OwningOpRef<cir::AllocaOp> varPtrOp =
        cir::AllocaOp::create(b, loc, structPptrTy, "S", getAlignOne(&context));
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

TEST_F(CIROpenACCPointerLikeTest, testPointerToComplex) {
  testScalarType(cir::ComplexType::get(cir::SingleType::get(&context)));
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
  mlir::Type vecTy = cir::VectorType::getFixed(i32Ty, 4);
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
  cir::RecordType structTy =
      cir::StructType::get(&context, getUniqueRecordName("S"),
                           /*is_class=*/false);
  mlir::Type members[] = {i32Ty, i32Ty};
  structTy.complete(members, /*packed=*/false, /*padding=*/{},
                    cir::RecordType::getAllDataKinds(members));
  testPointerToMemberType(structTy, mlir::acc::VariableTypeCategory::composite);
}

TEST_F(CIROpenACCPointerLikeTest, testPointerCast) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32Ty = cir::IntType::get(&context, 32, true);
  Type ptrTy = cir::PointerType::get(i32Ty);
  Value host = cir::AllocaOp::create(b, loc, ptrTy, "n", getAlignOne(&context));
  auto pointerLike = cast<mlir::acc::PointerLikeType>(ptrTy);

  Type i8Ptr = cir::PointerType::get(cir::IntType::get(&context, 8, true));
  Value asI8Ptr = pointerLike.genCast(b, loc, host, i8Ptr);
  ASSERT_TRUE(asI8Ptr);
  EXPECT_EQ(asI8Ptr.getDefiningOp<cir::CastOp>().getKind(),
            cir::CastKind::bitcast);

  auto globalSpace = cir::LangAddressSpaceAttr::get(
      &context, cir::LangAddressSpace::OffloadGlobal);
  Type globalPtr = cir::PointerType::get(i32Ty, globalSpace);
  Value asGlobal = pointerLike.genCast(b, loc, host, globalPtr);
  ASSERT_TRUE(asGlobal);
  EXPECT_EQ(asGlobal.getDefiningOp<cir::CastOp>().getKind(),
            cir::CastKind::address_space);

  Value loaded = pointerLike.genLoad(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(host), i32Ty);
  ASSERT_TRUE(loaded);
  Value asPtr = pointerLike.genCast(b, loc, loaded, ptrTy);
  ASSERT_TRUE(asPtr);
  EXPECT_EQ(asPtr.getDefiningOp<cir::CastOp>().getKind(),
            cir::CastKind::int_to_ptr);

  Type i64Ty = cir::IntType::get(&context, 64, true);
  // Pointer-to-integer produces no cast because that conversion is not
  // implemented. This records the current gap; update the check when support
  // is added.
  EXPECT_FALSE(pointerLike.genCast(b, loc, host, i64Ty));
  // Value-to-value casts are not produced here.
  EXPECT_FALSE(pointerLike.genCast(b, loc, loaded, i64Ty));
}

// A pointee with no size cannot be allocated or copied.
TEST_F(CIROpenACCPointerLikeTest, testUnsizedPointeeIsNotAllocated) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());
  Type voidPtr = cir::PointerType::get(cir::VoidType::get(&context));
  auto pointerLike = cast<mlir::acc::PointerLikeType>(voidPtr);
  bool needsFree = false;
  Value host =
      cir::AllocaOp::create(b, loc, voidPtr, "p", getAlignOne(&context));
  EXPECT_FALSE(pointerLike.genAllocate(b, loc, "p", voidPtr, host, needsFree));
  EXPECT_FALSE(pointerLike.genCopy(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(host),
      cast<TypedValue<mlir::acc::PointerLikeType>>(host), voidPtr));
}

TEST_F(CIROpenACCPointerLikeTest, testOffloadAddressSpace) {
  Type i32Ty = cir::IntType::get(&context, 32, true);
  auto global = cir::LangAddressSpaceAttr::get(
      &context, cir::LangAddressSpace::OffloadGlobal);
  auto generic = cir::LangAddressSpaceAttr::get(
      &context, cir::LangAddressSpace::OffloadGeneric);
  Type globalPtr = cir::PointerType::get(i32Ty, global);
  Type genericPtr = cir::PointerType::get(i32Ty, generic);
  Type hostPtr = cir::PointerType::get(i32Ty);

  OwningOpRef<cir::AllocaOp> globalAlloca =
      cir::AllocaOp::create(b, loc, globalPtr, "g", getAlignOne(&context));
  OwningOpRef<cir::AllocaOp> genericAlloca =
      cir::AllocaOp::create(b, loc, genericPtr, "n", getAlignOne(&context));
  OwningOpRef<cir::AllocaOp> hostAlloca =
      cir::AllocaOp::create(b, loc, hostPtr, "h", getAlignOne(&context));

  auto hostShared = cir::LangAddressSpaceAttr::get(
      &context, cir::LangAddressSpace::OffloadGlobalHost);
  Type hostSharedPtr = cir::PointerType::get(i32Ty, hostShared);
  OwningOpRef<cir::AllocaOp> hostSharedAlloca =
      cir::AllocaOp::create(b, loc, hostSharedPtr, "hs", getAlignOne(&context));

  auto globalLike = cast<mlir::acc::PointerLikeType>(globalPtr);
  auto genericLike = cast<mlir::acc::PointerLikeType>(genericPtr);
  auto hostLike = cast<mlir::acc::PointerLikeType>(hostPtr);
  auto hostSharedLike = cast<mlir::acc::PointerLikeType>(hostSharedPtr);
  EXPECT_TRUE(globalLike.isDeviceAccessible(globalAlloca.get(), nullptr));
  EXPECT_TRUE(globalLike.isInDeviceMemory(globalAlloca.get(), nullptr));
  EXPECT_TRUE(genericLike.isDeviceAccessible(genericAlloca.get(), nullptr));
  EXPECT_FALSE(genericLike.isInDeviceMemory(genericAlloca.get(), nullptr));
  EXPECT_TRUE(
      hostSharedLike.isDeviceAccessible(hostSharedAlloca.get(), nullptr));
  EXPECT_FALSE(
      hostSharedLike.isInDeviceMemory(hostSharedAlloca.get(), nullptr));
  EXPECT_FALSE(hostLike.isDeviceAccessible(hostAlloca.get(), nullptr));
  EXPECT_FALSE(hostLike.isInDeviceMemory(hostAlloca.get(), nullptr));
}

TEST_F(CIROpenACCPointerLikeTest, testGetAsMemRefType) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Type f32 = cir::SingleType::get(&context);
  Type i32 = cir::IntType::get(&context, 32, true);

  auto asMemRef = [&](Type pointer) {
    return cast<mlir::acc::PointerLikeType>(pointer).getAsMemRefType(*module);
  };

  EXPECT_EQ(asMemRef(cir::PointerType::get(f32)),
            MemRefType::get({}, Float32Type::get(&context)));
  EXPECT_EQ(asMemRef(cir::PointerType::get(cir::BoolType::get(&context))),
            MemRefType::get({}, IntegerType::get(&context, 8)));

  EXPECT_EQ(asMemRef(cir::PointerType::get(cir::ArrayType::get(f32, 8))),
            MemRefType::get({8}, Float32Type::get(&context)));

  Type inner = cir::ArrayType::get(f32, 4);
  EXPECT_EQ(asMemRef(cir::PointerType::get(cir::ArrayType::get(inner, 3))),
            MemRefType::get({3, 4}, Float32Type::get(&context)));

  EXPECT_FALSE(asMemRef(cir::PointerType::get(cir::PointerType::get(f32))));
  EXPECT_FALSE(asMemRef(cir::PointerType::get(cir::VoidType::get(&context))));

  cir::RecordType structTy = cir::StructType::get(
      &context, getUniqueRecordName("Mem"), /*is_class=*/false);
  Type members[] = {i32};
  structTy.complete(members, /*packed=*/false, /*padding=*/{},
                    cir::RecordType::getAllDataKinds(members));
  EXPECT_FALSE(asMemRef(cir::PointerType::get(structTy)));
}

TEST_F(CIROpenACCPointerLikeTest, testDynamicArrayIsNotScalar) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32Ty = cir::IntType::get(&context, 32, true);
  Type ptrTy = cir::PointerType::get(i32Ty);
  Type countTy = cir::IntType::get(&context, 64, false);
  Value count = cir::ConstantOp::create(b, loc, cir::IntAttr::get(countTy, 4));
  Value vla =
      cir::AllocaOp::create(b, loc, ptrTy, "vla", getAlignOne(&context), count);

  auto pointerLike = cast<mlir::acc::PointerLikeType>(ptrTy);
  EXPECT_EQ(pointerLike.getPointeeTypeCategory(
                cast<TypedValue<mlir::acc::PointerLikeType>>(vla), i32Ty),
            mlir::acc::VariableTypeCategory::array);

  Type i8Ptr = cir::PointerType::get(cir::IntType::get(&context, 8, true));
  Value casted =
      cir::CastOp::create(b, loc, i8Ptr, cir::CastKind::bitcast, vla);
  EXPECT_EQ(cast<mlir::acc::PointerLikeType>(i8Ptr).getPointeeTypeCategory(
                cast<TypedValue<mlir::acc::PointerLikeType>>(casted), i32Ty),
            mlir::acc::VariableTypeCategory::array);

  Value index = cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32Ty, 1));
  Value element = cir::PtrStrideOp::create(b, loc, ptrTy, vla, index);
  EXPECT_EQ(pointerLike.getPointeeTypeCategory(
                cast<TypedValue<mlir::acc::PointerLikeType>>(element), i32Ty),
            mlir::acc::VariableTypeCategory::array);

  bool needsFree = true;
  Value privatized =
      pointerLike.genAllocate(b, loc, "vla", ptrTy, vla, needsFree);
  ASSERT_TRUE(privatized);
  auto privAlloca = privatized.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(privAlloca);
  EXPECT_TRUE(privAlloca.isDynamic());
  auto privCount =
      privAlloca.getDynAllocSize().getDefiningOp<cir::ConstantOp>();
  ASSERT_TRUE(privCount);
  EXPECT_EQ(privCount.getValue(),
            count.getDefiningOp<cir::ConstantOp>().getValue());
  EXPECT_EQ(privAlloca.getAllocaType(), i32Ty);
}

// When the pointer is not produced by an allocation, only its type describes
// the object. A scalar or a fixed-size array is complete in that type, so it
// is allocated. A type with no size cannot be.
TEST_F(CIROpenACCPointerLikeTest, testAllocateWithoutAlloca) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Type arrayPtr = cir::PointerType::get(cir::ArrayType::get(i32, 4));
  Type voidPtr = cir::PointerType::get(cir::VoidType::get(&context));

  auto allocateArg = [&](Type ptrTy, StringRef name) {
    b.setInsertionPointToStart(module->getBody());
    auto funcTy = cir::FuncType::get(SmallVector<Type, 1>{ptrTy},
                                     cir::VoidType::get(&context));
    cir::FuncOp func = cir::FuncOp::create(b, loc, name, funcTy);
    b.setInsertionPointToStart(func.addEntryBlock());
    bool needsFree = true;
    Value arg = func.getBody().front().getArgument(0);
    Value allocated = cast<mlir::acc::PointerLikeType>(ptrTy).genAllocate(
        b, loc, name, ptrTy, arg, needsFree);
    return std::pair<Value, bool>(allocated, needsFree);
  };

  auto [scalar, scalarNeedsFree] = allocateArg(i32Ptr, "scalar");
  ASSERT_TRUE(scalar);
  EXPECT_FALSE(scalarNeedsFree);
  auto scalarAlloca = scalar.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(scalarAlloca);
  EXPECT_FALSE(scalarAlloca.isDynamic());
  EXPECT_EQ(scalarAlloca.getAllocaType(), i32);

  auto [array, arrayNeedsFree] = allocateArg(arrayPtr, "array");
  ASSERT_TRUE(array);
  EXPECT_FALSE(arrayNeedsFree);
  auto arrayAlloca = array.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(arrayAlloca);
  EXPECT_EQ(arrayAlloca.getAllocaType(), cir::ArrayType::get(i32, 4));
  EXPECT_FALSE(arrayAlloca.isDynamic());

  auto [voidArg, voidNeedsFree] = allocateArg(voidPtr, "voidp");
  EXPECT_FALSE(voidArg);

  // A loaded pointer describes the object it points to, not the slot it was
  // loaded from.
  b.setInsertionPointToStart(module->getBody());
  Type ptrPtr = cir::PointerType::get(i32Ptr);
  Value slot =
      cir::AllocaOp::create(b, loc, ptrPtr, "slot", getAlignOne(&context));
  Value loaded = cir::LoadOp::create(b, loc, slot);
  bool needsFree = true;
  Value fromLoad = cast<mlir::acc::PointerLikeType>(i32Ptr).genAllocate(
      b, loc, "loaded", i32Ptr, loaded, needsFree);
  ASSERT_TRUE(fromLoad);
  EXPECT_FALSE(needsFree);
  auto loadedAlloca = fromLoad.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(loadedAlloca);
  EXPECT_EQ(loadedAlloca.getAllocaType(), i32);
  EXPECT_FALSE(loadedAlloca.isDynamic());
}

// A conversion that still names the whole allocation allocates that object,
// and a constant element count is part of its extent. An offset or a member
// names only part of the allocation, so the whole object is not allocated.
TEST_F(CIROpenACCPointerLikeTest, testAllocateThroughPointerViews) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i8 = cir::IntType::get(&context, 8, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Type i8Ptr = cir::PointerType::get(i8);
  Type arrayTy = cir::ArrayType::get(i32, 4);
  Type arrayPtr = cir::PointerType::get(arrayTy);
  IntegerAttr align = getAlignOne(&context);

  auto allocate = [&](Value view) {
    bool needsFree = true;
    Value allocated =
        cast<mlir::acc::PointerLikeType>(view.getType())
            .genAllocate(b, loc, "priv", view.getType(), view, needsFree);
    if (allocated)
      EXPECT_FALSE(needsFree);
    return allocated;
  };

  Value scalar = cir::AllocaOp::create(b, loc, i32Ptr, "scalar", align);
  Value scalarCast =
      cir::CastOp::create(b, loc, i8Ptr, cir::CastKind::bitcast, scalar);
  Value scalarStride = cir::PtrStrideOp::create(
      b, loc, i32Ptr, scalar,
      cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32, 1)));

  Value castAlloc = allocate(scalarCast);
  auto castOp = castAlloc.getDefiningOp<cir::CastOp>();
  ASSERT_TRUE(castOp);
  EXPECT_EQ(castOp.getKind(), cir::CastKind::bitcast);
  auto castBase = castOp.getSrc().getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(castBase);
  EXPECT_EQ(castBase.getAllocaType(), i32);
  EXPECT_FALSE(castBase.isDynamic());

  EXPECT_FALSE(allocate(scalarStride));

  Value array = cir::AllocaOp::create(b, loc, arrayPtr, "array", align);
  Value decay = cir::CastOp::create(b, loc, i32Ptr,
                                    cir::CastKind::array_to_ptrdecay, array);
  Value element = cir::PtrStrideOp::create(
      b, loc, i32Ptr, decay,
      cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32, 2)));
  Value decayedByte =
      cir::CastOp::create(b, loc, i8Ptr, cir::CastKind::bitcast, element);

  EXPECT_FALSE(allocate(element));
  EXPECT_FALSE(allocate(decayedByte));

  Type countTy = cir::IntType::get(&context, 64, false);
  Value count = cir::ConstantOp::create(b, loc, cir::IntAttr::get(countTy, 4));
  Value vla = cir::AllocaOp::create(b, loc, i32Ptr, "vla", align, count);
  Value vlaCast =
      cir::CastOp::create(b, loc, i8Ptr, cir::CastKind::bitcast, vla);
  Value vlaElement = cir::PtrStrideOp::create(
      b, loc, i32Ptr, vla,
      cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32, 1)));

  Value vlaCastAlloc = allocate(vlaCast);
  auto vlaCastOp = vlaCastAlloc.getDefiningOp<cir::CastOp>();
  ASSERT_TRUE(vlaCastOp);
  auto vlaCastBase = vlaCastOp.getSrc().getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(vlaCastBase);
  EXPECT_TRUE(vlaCastBase.isDynamic());
  auto vlaCastCount =
      vlaCastBase.getDynAllocSize().getDefiningOp<cir::ConstantOp>();
  ASSERT_TRUE(vlaCastCount);
  EXPECT_EQ(vlaCastCount.getValue(),
            count.getDefiningOp<cir::ConstantOp>().getValue());
  EXPECT_EQ(vlaCastBase.getAllocaType(), i32);

  EXPECT_FALSE(allocate(vlaElement));

  cir::RecordType record =
      cir::StructType::get(&context, getUniqueRecordName("Point"),
                           /*is_class=*/false);
  Type fields[] = {i32, arrayTy};
  record.complete(fields, /*packed=*/false, /*padding=*/{},
                  cir::RecordType::getAllDataKinds(fields));
  Type recordPtr = cir::PointerType::get(record);
  Value object = cir::AllocaOp::create(b, loc, recordPtr, "object", align);
  Value field = cir::GetMemberOp::create(b, loc, i32Ptr, object, "x", 0u);
  Value arrayField =
      cir::GetMemberOp::create(b, loc, arrayPtr, object, "a", 1u);
  EXPECT_FALSE(allocate(field));
  EXPECT_FALSE(allocate(arrayField));
}

// Decaying an array to a pointer does not change the allocated object, so
// the copy covers the array.
TEST_F(CIROpenACCPointerLikeTest, testCopyCoversAllocatedObject) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Type arrayTy = cir::ArrayType::get(i32, 4);
  Type arrayPtr = cir::PointerType::get(arrayTy);
  auto pointerLike = cast<mlir::acc::PointerLikeType>(i32Ptr);

  Value array =
      cir::AllocaOp::create(b, loc, arrayPtr, "array", getAlignOne(&context));
  Value decay = cir::CastOp::create(b, loc, i32Ptr,
                                    cir::CastKind::array_to_ptrdecay, array);
  bool needsFree = true;
  Value privatized =
      pointerLike.genAllocate(b, loc, "array", i32Ptr, decay, needsFree);
  ASSERT_TRUE(privatized);
  EXPECT_TRUE(pointerLike.genCopy(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(privatized),
      cast<TypedValue<mlir::acc::PointerLikeType>>(decay), i32Ptr));
  cir::CopyOp copy;
  module->walk([&](cir::CopyOp op) { copy = op; });
  ASSERT_TRUE(copy);
  EXPECT_EQ(cast<cir::PointerType>(copy.getSrc().getType()).getPointee(),
            arrayTy);
}

// An offset of zero still names the allocated object, so that object is
// allocated.
TEST_F(CIROpenACCPointerLikeTest, testZeroStrideAllocatesBase) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Value scalar =
      cir::AllocaOp::create(b, loc, i32Ptr, "scalar", getAlignOne(&context));
  Value zero = cir::ConstantOp::create(b, loc, cir::IntAttr::get(i32, 0));
  Value view = cir::PtrStrideOp::create(b, loc, i32Ptr, scalar, zero);

  bool needsFree = true;
  Value allocated = cast<mlir::acc::PointerLikeType>(i32Ptr).genAllocate(
      b, loc, "priv", i32Ptr, view, needsFree);
  ASSERT_TRUE(allocated);
  EXPECT_FALSE(needsFree);
  auto alloca = allocated.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(alloca);
  EXPECT_EQ(alloca.getAllocaType(), i32);
  EXPECT_FALSE(alloca.isDynamic());
}

// The number of elements is known only at run time, so it is not part of the
// type. The amount copied comes from the type, so this object cannot be
// copied. Allocating an object of runtime size is not supported.
TEST_F(CIROpenACCPointerLikeTest, testRuntimeSizeIsNotAllocated) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Type countTy = cir::IntType::get(&context, 64, false);
  Type countPtr = cir::PointerType::get(countTy);
  IntegerAttr align = getAlignOne(&context);

  Value countSlot = cir::AllocaOp::create(b, loc, countPtr, "n.addr", align);
  Value count = cir::LoadOp::create(b, loc, countSlot);
  Value vla = cir::AllocaOp::create(b, loc, i32Ptr, "vla", align, count);
  Value dst = cir::AllocaOp::create(b, loc, i32Ptr, "dst", align);

  auto pointerLike = cast<mlir::acc::PointerLikeType>(i32Ptr);
  bool needsFree = true;
  EXPECT_FALSE(pointerLike.genAllocate(b, loc, "vla", i32Ptr, vla, needsFree));
  EXPECT_FALSE(pointerLike.genCopy(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(dst),
      cast<TypedValue<mlir::acc::PointerLikeType>>(vla), i32Ptr));
}

// The element count is a constant, so the object is the whole array and the
// copy covers it.
TEST_F(CIROpenACCPointerLikeTest, testConstantSizeCopyCoversArray) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Type countTy = cir::IntType::get(&context, 64, false);
  Type arrayTy = cir::ArrayType::get(i32, 4);
  IntegerAttr align = getAlignOne(&context);

  Value count = cir::ConstantOp::create(b, loc, cir::IntAttr::get(countTy, 4));
  Value src = cir::AllocaOp::create(b, loc, i32Ptr, "src", align, count);
  Value dst = cir::AllocaOp::create(b, loc, i32Ptr, "dst", align, count);

  auto pointerLike = cast<mlir::acc::PointerLikeType>(i32Ptr);
  EXPECT_TRUE(pointerLike.genCopy(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(dst),
      cast<TypedValue<mlir::acc::PointerLikeType>>(src), i32Ptr));

  cir::CopyOp copy;
  module->walk([&](cir::CopyOp op) { copy = op; });
  ASSERT_TRUE(copy);
  EXPECT_EQ(cast<cir::PointerType>(copy.getSrc().getType()).getPointee(),
            arrayTy);
  EXPECT_EQ(cast<cir::PointerType>(copy.getDst().getType()).getPointee(),
            arrayTy);
}

// Moving the pointer to another address space does not change the object.
// The allocated pointer remains in that address space.
TEST_F(CIROpenACCPointerLikeTest, testAddressSpaceCastKeepsAddressSpace) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type hostPtr = cir::PointerType::get(i32);
  auto global = cir::LangAddressSpaceAttr::get(
      &context, cir::LangAddressSpace::OffloadGlobal);
  Type globalPtr = cir::PointerType::get(i32, global);

  Value host =
      cir::AllocaOp::create(b, loc, hostPtr, "h", getAlignOne(&context));
  Value view = cir::CastOp::create(b, loc, globalPtr,
                                   cir::CastKind::address_space, host);

  bool needsFree = true;
  Value allocated = cast<mlir::acc::PointerLikeType>(globalPtr).genAllocate(
      b, loc, "priv", globalPtr, view, needsFree);
  ASSERT_TRUE(allocated);
  EXPECT_FALSE(needsFree);
  EXPECT_EQ(allocated.getType(), globalPtr);
  auto alloca = allocated.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(alloca);
  EXPECT_EQ(alloca.getAllocaType(), i32);
}

// The allocated object is the pointer. The memory it refers to belongs to a
// different object.
TEST_F(CIROpenACCPointerLikeTest, testPointerObjectAllocatesOnePointer) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Type ptrPtr = cir::PointerType::get(i32Ptr);
  Value slot =
      cir::AllocaOp::create(b, loc, ptrPtr, "p", getAlignOne(&context));

  bool needsFree = true;
  Value allocated = cast<mlir::acc::PointerLikeType>(ptrPtr).genAllocate(
      b, loc, "priv", ptrPtr, slot, needsFree);
  ASSERT_TRUE(allocated);
  EXPECT_FALSE(needsFree);
  auto alloca = allocated.getDefiningOp<cir::AllocaOp>();
  ASSERT_TRUE(alloca);
  EXPECT_EQ(alloca.getAllocaType(), i32Ptr);
  EXPECT_FALSE(alloca.isDynamic());
}

// Records, including a constant number of them, are an implementation
// limitation. This test anchors that decision so it can be updated if
// support is added.
TEST_F(CIROpenACCPointerLikeTest, testRecordObjectIsNotAllocated) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type countTy = cir::IntType::get(&context, 64, false);
  cir::RecordType record =
      cir::StructType::get(&context, getUniqueRecordName("S"),
                           /*is_class=*/false);
  Type members[] = {i32, i32};
  record.complete(members, /*packed=*/false, /*padding=*/{},
                  cir::RecordType::getAllDataKinds(members));
  Type recordPtr = cir::PointerType::get(record);
  IntegerAttr align = getAlignOne(&context);

  Value plain = cir::AllocaOp::create(b, loc, recordPtr, "s", align);
  Value count = cir::ConstantOp::create(b, loc, cir::IntAttr::get(countTy, 4));
  Value sized = cir::AllocaOp::create(b, loc, recordPtr, "arr", align, count);

  auto pointerLike = cast<mlir::acc::PointerLikeType>(recordPtr);
  bool needsFree = true;
  EXPECT_FALSE(
      pointerLike.genAllocate(b, loc, "s", recordPtr, plain, needsFree));
  EXPECT_FALSE(
      pointerLike.genAllocate(b, loc, "arr", recordPtr, sized, needsFree));
  EXPECT_FALSE(pointerLike.genCopy(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(plain),
      cast<TypedValue<mlir::acc::PointerLikeType>>(sized), recordPtr));
  EXPECT_FALSE(pointerLike.genFree(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(plain), Value(),
      recordPtr));
}

// An object that can be allocated can be freed, even if there is nothing to
// free.
TEST_F(CIROpenACCPointerLikeTest, testFreeFollowsAllocaObject) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());

  Type i32 = cir::IntType::get(&context, 32, true);
  Type i32Ptr = cir::PointerType::get(i32);
  Value scalar =
      cir::AllocaOp::create(b, loc, i32Ptr, "scalar", getAlignOne(&context));

  EXPECT_TRUE(cast<mlir::acc::PointerLikeType>(i32Ptr).genFree(
      b, loc, cast<TypedValue<mlir::acc::PointerLikeType>>(scalar), Value(),
      i32Ptr));
}
