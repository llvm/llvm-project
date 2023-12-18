//===- FIRTypesTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct FIRTypesTest : public testing::Test {
public:
  void SetUp() {
    fir::support::loadDialects(context);
    kindMap = new fir::KindMapping(&context, kindMapInit, "r42a10c14d28i40l41");
  }
  mlir::MLIRContext context;
  fir::KindMapping *kindMap{};
  std::string kindMapInit =
      "i10:80,l3:24,a1:8,r54:Double,r62:X86_FP80,r11:PPC_FP128";
};

// Test fir::isPolymorphicType from flang/Optimizer/Dialect/FIRType.h.
TEST_F(FIRTypesTest, isPolymorphicTypeTest) {
  mlir::Type noneTy = mlir::NoneType::get(&context);
  mlir::Type seqNoneTy =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, noneTy);
  mlir::Type recTy = fir::RecordType::get(&context, "dt");
  mlir::Type seqRecTy =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, recTy);

  // CLASS(T)
  mlir::Type ty = fir::ClassType::get(recTy);
  EXPECT_TRUE(fir::isPolymorphicType(ty));
  EXPECT_TRUE(fir::isPolymorphicType(fir::ReferenceType::get(ty)));

  // CLASS(T), DIMENSION(10)
  ty = fir::ClassType::get(fir::SequenceType::get({10}, recTy));
  EXPECT_TRUE(fir::isPolymorphicType(ty));

  // CLASS(T), DIMENSION(:)
  ty = fir::ClassType::get(seqRecTy);
  EXPECT_TRUE(fir::isPolymorphicType(ty));

  // CLASS(T), ALLOCATABLE
  ty = fir::ClassType::get(fir::HeapType::get(recTy));
  EXPECT_TRUE(fir::isPolymorphicType(ty));

  // CLASS(T), ALLOCATABLE, DIMENSION(:)
  ty = fir::ClassType::get(fir::HeapType::get(seqRecTy));
  EXPECT_TRUE(fir::isPolymorphicType(ty));

  // CLASS(T), POINTER
  ty = fir::ClassType::get(fir::PointerType::get(recTy));
  EXPECT_TRUE(fir::isPolymorphicType(ty));

  // CLASS(T), POINTER, DIMENSIONS(:)
  ty = fir::ClassType::get(fir::PointerType::get(seqRecTy));
  EXPECT_TRUE(fir::isPolymorphicType(ty));

  // CLASS(*)
  ty = fir::ClassType::get(noneTy);
  EXPECT_TRUE(fir::isPolymorphicType(ty));
  EXPECT_TRUE(fir::isPolymorphicType(fir::ReferenceType::get(ty)));

  // TYPE(*)
  EXPECT_TRUE(fir::isPolymorphicType(fir::BoxType::get(noneTy)));

  // TYPE(*), DIMENSION(:)
  EXPECT_TRUE(fir::isPolymorphicType(fir::BoxType::get(seqNoneTy)));

  // false tests
  EXPECT_FALSE(fir::isPolymorphicType(noneTy));
  EXPECT_FALSE(fir::isPolymorphicType(seqNoneTy));
}

// Test fir::isUnlimitedPolymorphicType from flang/Optimizer/Dialect/FIRType.h.
TEST_F(FIRTypesTest, isUnlimitedPolymorphicTypeTest) {
  mlir::Type noneTy = mlir::NoneType::get(&context);
  mlir::Type seqNoneTy =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, noneTy);

  // CLASS(*)
  mlir::Type ty = fir::ClassType::get(noneTy);
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(fir::ReferenceType::get(ty)));

  // CLASS(*), DIMENSION(10)
  ty = fir::ClassType::get(fir::SequenceType::get({10}, noneTy));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));

  // CLASS(*), DIMENSION(:)
  ty = fir::ClassType::get(
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, noneTy));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));

  // CLASS(*), ALLOCATABLE
  ty = fir::ClassType::get(fir::HeapType::get(noneTy));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));

  // CLASS(*), ALLOCATABLE, DIMENSION(:)
  ty = fir::ClassType::get(fir::HeapType::get(seqNoneTy));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));

  // CLASS(*), POINTER
  ty = fir::ClassType::get(fir::PointerType::get(noneTy));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));

  // CLASS(*), POINTER, DIMENSIONS(:)
  ty = fir::ClassType::get(fir::PointerType::get(seqNoneTy));
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(ty));

  // TYPE(*)
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(fir::BoxType::get(noneTy)));

  // TYPE(*), DIMENSION(:)
  EXPECT_TRUE(fir::isUnlimitedPolymorphicType(fir::BoxType::get(seqNoneTy)));

  // false tests
  EXPECT_FALSE(fir::isUnlimitedPolymorphicType(noneTy));
  EXPECT_FALSE(fir::isUnlimitedPolymorphicType(seqNoneTy));
}

// Test fir::isBoxedRecordType from flang/Optimizer/Dialect/FIRType.h.
TEST_F(FIRTypesTest, isBoxedRecordType) {
  mlir::Type recTy = fir::RecordType::get(&context, "dt");
  mlir::Type seqRecTy =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, recTy);
  mlir::Type ty = fir::BoxType::get(recTy);
  EXPECT_TRUE(fir::isBoxedRecordType(ty));
  EXPECT_TRUE(fir::isBoxedRecordType(fir::ReferenceType::get(ty)));

  // TYPE(T), ALLOCATABLE
  ty = fir::BoxType::get(fir::HeapType::get(recTy));
  EXPECT_TRUE(fir::isBoxedRecordType(ty));

  // TYPE(T), POINTER
  ty = fir::BoxType::get(fir::PointerType::get(recTy));
  EXPECT_TRUE(fir::isBoxedRecordType(ty));

  // TYPE(T), DIMENSION(10)
  ty = fir::BoxType::get(fir::SequenceType::get({10}, recTy));
  EXPECT_TRUE(fir::isBoxedRecordType(ty));

  // TYPE(T), DIMENSION(:)
  ty = fir::BoxType::get(seqRecTy);
  EXPECT_TRUE(fir::isBoxedRecordType(ty));

  EXPECT_FALSE(fir::isBoxedRecordType(fir::BoxType::get(
      fir::ReferenceType::get(mlir::IntegerType::get(&context, 32)))));
}

// Test fir::isScalarBoxedRecordType from flang/Optimizer/Dialect/FIRType.h.
TEST_F(FIRTypesTest, isScalarBoxedRecordType) {
  mlir::Type recTy = fir::RecordType::get(&context, "dt");
  mlir::Type seqRecTy =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, recTy);
  mlir::Type ty = fir::BoxType::get(recTy);
  EXPECT_TRUE(fir::isScalarBoxedRecordType(ty));
  EXPECT_TRUE(fir::isScalarBoxedRecordType(fir::ReferenceType::get(ty)));

  // CLASS(T), ALLOCATABLE
  ty = fir::ClassType::get(fir::HeapType::get(recTy));
  EXPECT_TRUE(fir::isScalarBoxedRecordType(ty));

  // TYPE(T), ALLOCATABLE
  ty = fir::BoxType::get(fir::HeapType::get(recTy));
  EXPECT_TRUE(fir::isScalarBoxedRecordType(ty));

  // TYPE(T), POINTER
  ty = fir::BoxType::get(fir::PointerType::get(recTy));
  EXPECT_TRUE(fir::isScalarBoxedRecordType(ty));

  // CLASS(T), POINTER
  ty = fir::ClassType::get(fir::PointerType::get(recTy));
  EXPECT_TRUE(fir::isScalarBoxedRecordType(ty));

  // TYPE(T), DIMENSION(10)
  ty = fir::BoxType::get(fir::SequenceType::get({10}, recTy));
  EXPECT_FALSE(fir::isScalarBoxedRecordType(ty));

  // TYPE(T), DIMENSION(:)
  ty = fir::BoxType::get(seqRecTy);
  EXPECT_FALSE(fir::isScalarBoxedRecordType(ty));

  EXPECT_FALSE(fir::isScalarBoxedRecordType(fir::BoxType::get(
      fir::ReferenceType::get(mlir::IntegerType::get(&context, 32)))));
}

TEST_F(FIRTypesTest, updateTypeForUnlimitedPolymorphic) {
  // RecordType are not changed.

  // !fir.tyep<T> -> !fir.type<T>
  mlir::Type recTy = fir::RecordType::get(&context, "dt");
  EXPECT_EQ(recTy, fir::updateTypeForUnlimitedPolymorphic(recTy));

  // !fir.array<2x!fir.type<T>> -> !fir.array<2x!fir.type<T>>
  mlir::Type arrRecTy = fir::SequenceType::get({2}, recTy);
  EXPECT_EQ(arrRecTy, fir::updateTypeForUnlimitedPolymorphic(arrRecTy));

  // !fir.heap<!fir.type<T>> -> !fir.heap<!fir.type<T>>
  mlir::Type heapTy = fir::HeapType::get(recTy);
  EXPECT_EQ(heapTy, fir::updateTypeForUnlimitedPolymorphic(heapTy));
  // !fir.heap<!fir.array<2x!fir.type<T>>> ->
  // !fir.heap<!fir.array<2x!fir.type<T>>>
  mlir::Type heapArrTy = fir::HeapType::get(arrRecTy);
  EXPECT_EQ(heapArrTy, fir::updateTypeForUnlimitedPolymorphic(heapArrTy));

  // !fir.ptr<!fir.type<T>> -> !fir.ptr<!fir.type<T>>
  mlir::Type ptrTy = fir::PointerType::get(recTy);
  EXPECT_EQ(ptrTy, fir::updateTypeForUnlimitedPolymorphic(ptrTy));
  // !fir.ptr<!fir.array<2x!fir.type<T>>> ->
  // !fir.ptr<!fir.array<2x!fir.type<T>>>
  mlir::Type ptrArrTy = fir::PointerType::get(arrRecTy);
  EXPECT_EQ(ptrArrTy, fir::updateTypeForUnlimitedPolymorphic(ptrArrTy));

  // When updating intrinsic types the array, pointer and heap types are kept.
  // only the inner element type is changed to `none`.
  mlir::Type none = mlir::NoneType::get(&context);
  mlir::Type arrNone = fir::SequenceType::get({10}, none);
  mlir::Type heapNone = fir::HeapType::get(none);
  mlir::Type heapArrNone = fir::HeapType::get(arrNone);
  mlir::Type ptrNone = fir::PointerType::get(none);
  mlir::Type ptrArrNone = fir::PointerType::get(arrNone);

  mlir::Type i32Ty = mlir::IntegerType::get(&context, 32);
  mlir::Type f32Ty = mlir::FloatType::getF32(&context);
  mlir::Type l1Ty = fir::LogicalType::get(&context, 1);
  mlir::Type cplx4Ty = fir::ComplexType::get(&context, 4);
  mlir::Type char1Ty = fir::CharacterType::get(&context, 1, 10);
  llvm::SmallVector<mlir::Type> intrinsicTypes = {
      i32Ty, f32Ty, l1Ty, cplx4Ty, char1Ty};

  for (mlir::Type ty : intrinsicTypes) {
    // `ty` -> none
    EXPECT_EQ(none, fir::updateTypeForUnlimitedPolymorphic(ty));

    // !fir.array<10xTY> -> !fir.array<10xnone>
    mlir::Type arrTy = fir::SequenceType::get({10}, ty);
    EXPECT_EQ(arrNone, fir::updateTypeForUnlimitedPolymorphic(arrTy));

    // !fir.heap<TY> -> !fir.heap<none>
    mlir::Type heapTy = fir::HeapType::get(ty);
    EXPECT_EQ(heapNone, fir::updateTypeForUnlimitedPolymorphic(heapTy));

    // !fir.heap<!fir.array<10xTY>> -> !fir.heap<!fir.array<10xnone>>
    mlir::Type heapArrTy = fir::HeapType::get(arrTy);
    EXPECT_EQ(heapArrNone, fir::updateTypeForUnlimitedPolymorphic(heapArrTy));

    // !fir.ptr<TY> -> !fir.ptr<none>
    mlir::Type ptrTy = fir::PointerType::get(ty);
    EXPECT_EQ(ptrNone, fir::updateTypeForUnlimitedPolymorphic(ptrTy));

    // !fir.ptr<!fir.array<10xTY>> -> !fir.ptr<!fir.array<10xnone>>
    mlir::Type ptrArrTy = fir::PointerType::get(arrTy);
    EXPECT_EQ(ptrArrNone, fir::updateTypeForUnlimitedPolymorphic(ptrArrTy));
  }
}

TEST_F(FIRTypesTest, getTypeAsString) {
  EXPECT_EQ("i32",
      fir::getTypeAsString(mlir::IntegerType::get(&context, 32), *kindMap));
  EXPECT_EQ("ref_i32",
      fir::getTypeAsString(
          fir::ReferenceType::get(mlir::IntegerType::get(&context, 32)),
          *kindMap));
  EXPECT_EQ(
      "f64", fir::getTypeAsString(mlir::FloatType::getF64(&context), *kindMap));
  EXPECT_EQ(
      "l8", fir::getTypeAsString(fir::LogicalType::get(&context, 1), *kindMap));
  EXPECT_EQ("z32",
      fir::getTypeAsString(
          mlir::ComplexType::get(mlir::FloatType::getF32(&context)), *kindMap));
  EXPECT_EQ("c8",
      fir::getTypeAsString(fir::CharacterType::get(&context, 1, 1), *kindMap));
  EXPECT_EQ("c8x10",
      fir::getTypeAsString(fir::CharacterType::get(&context, 1, 10), *kindMap));
  mlir::Type ty = mlir::IntegerType::get(&context, 64);
  mlir::Type arrTy = fir::SequenceType::get({10, 20}, ty);
  EXPECT_EQ("10x20xi64", fir::getTypeAsString(arrTy, *kindMap));
  EXPECT_EQ(
      "idx", fir::getTypeAsString(mlir::IndexType::get(&context), *kindMap));
  EXPECT_EQ("ptr_i32",
      fir::getTypeAsString(
          fir::PointerType::get(mlir::IntegerType::get(&context, 32)),
          *kindMap));
  EXPECT_EQ("heap_i32",
      fir::getTypeAsString(
          fir::HeapType::get(mlir::IntegerType::get(&context, 32)), *kindMap));
  EXPECT_EQ("box_i32",
      fir::getTypeAsString(
          fir::BoxType::get(mlir::IntegerType::get(&context, 32)), *kindMap));
  EXPECT_EQ("class_i32",
      fir::getTypeAsString(
          fir::ClassType::get(mlir::IntegerType::get(&context, 32)), *kindMap));
  EXPECT_EQ("class_none",
      fir::getTypeAsString(
          fir::ClassType::get(mlir::NoneType::get(&context)), *kindMap));
  auto derivedTy = fir::RecordType::get(&context, "derived");
  llvm::SmallVector<std::pair<std::string, mlir::Type>> components;
  components.emplace_back("p1", mlir::IntegerType::get(&context, 64));
  derivedTy.finalize({}, components);
  EXPECT_EQ("rec_derived", fir::getTypeAsString(derivedTy, *kindMap));
  mlir::Type dynArrTy =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent(),
                                 fir::SequenceType::getUnknownExtent()},
          ty);
  EXPECT_EQ("UxUxi64", fir::getTypeAsString(dynArrTy, *kindMap));
  EXPECT_EQ("llvmptr_i32",
      fir::getTypeAsString(
          fir::LLVMPointerType::get(mlir::IntegerType::get(&context, 32)),
          *kindMap));
  EXPECT_EQ("boxchar_c8xU",
      fir::getTypeAsString(fir::BoxCharType::get(&context, 1), *kindMap));
}
