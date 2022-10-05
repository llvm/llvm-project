//===- FIRTypesTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct FIRTypesTest : public testing::Test {
public:
  void SetUp() { fir::support::loadDialects(context); }

  mlir::MLIRContext context;
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
