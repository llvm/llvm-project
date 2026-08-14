//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for UnionType: getTypeSizeInBits and isLayoutIdentical.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

class UnionTypeSizeTest : public ::testing::Test {
protected:
  UnionTypeSizeTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;

  mlir::StringAttr getName(llvm::StringRef name) {
    return mlir::StringAttr::get(&context, name);
  }
};

TEST_F(UnionTypeSizeTest, SizeInBitsNotBytes) {
  IntType i32 = IntType::get(&context, 32, true);
  auto ty = UnionType::get(&context, getName("U"));
  mlir::Type members[] = {i32};
  ty.complete(members, /*packed=*/false, /*padding=*/mlir::Type{},
              RecordType::getAllDataKinds(members));

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  mlir::DataLayout dl(module);

  llvm::TypeSize size = dl.getTypeSizeInBits(ty);
  EXPECT_EQ(size.getFixedValue(), 32u);

  module->erase();
}

TEST_F(UnionTypeSizeTest, MultiMemberUnion) {
  IntType i32 = IntType::get(&context, 32, true);
  IntType i64 = IntType::get(&context, 64, true);
  auto ty = UnionType::get(&context, getName("U2"));
  mlir::Type members[] = {i32, i64};
  ty.complete(members, /*packed=*/false, /*padding=*/mlir::Type{},
              RecordType::getAllDataKinds(members));

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  mlir::DataLayout dl(module);

  llvm::TypeSize size = dl.getTypeSizeInBits(ty);
  EXPECT_EQ(size.getFixedValue(), 64u);

  module->erase();
}

TEST_F(UnionTypeSizeTest, EmptyUnion) {
  auto ty = UnionType::get(&context, getName("Empty"));
  ty.complete(/*members=*/{}, /*packed=*/false, /*padding=*/mlir::Type{},
              /*memberKinds=*/{});

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  mlir::DataLayout dl(module);

  llvm::TypeSize size = dl.getTypeSizeInBits(ty);
  EXPECT_EQ(size.getFixedValue(), 0u);

  module->erase();
}

TEST_F(UnionTypeSizeTest, IsLayoutIdenticalNoPadding) {
  IntType i32 = IntType::get(&context, 32, true);
  mlir::Type members[] = {i32};
  auto ty1 = UnionType::get(&context, getName("Ua"));
  ty1.complete(members, /*packed=*/false, /*padding=*/mlir::Type{},
               RecordType::getAllDataKinds(members));
  auto ty2 = UnionType::get(&context, getName("Ub"));
  ty2.complete(members, /*packed=*/false, /*padding=*/mlir::Type{},
               RecordType::getAllDataKinds(members));
  EXPECT_TRUE(ty1.isLayoutIdentical(ty2));
}

TEST_F(UnionTypeSizeTest, IsLayoutIdenticalDifferentPadding) {
  IntType i32 = IntType::get(&context, 32, true);
  IntType i8 = IntType::get(&context, 8, false);
  IntType i16 = IntType::get(&context, 16, false);
  mlir::Type members[] = {i32};
  auto ty1 = UnionType::get(&context, getName("Upad1"));
  ty1.complete(members, /*packed=*/false, /*padding=*/i8,
               RecordType::getAllDataKinds(members));
  auto ty2 = UnionType::get(&context, getName("Upad2"));
  ty2.complete(members, /*packed=*/false, /*padding=*/i16,
               RecordType::getAllDataKinds(members));
  EXPECT_FALSE(ty1.isLayoutIdentical(ty2));
}

TEST_F(UnionTypeSizeTest, IsLayoutIdenticalSamePadding) {
  IntType i32 = IntType::get(&context, 32, true);
  IntType i8 = IntType::get(&context, 8, false);
  mlir::Type members[] = {i32};
  auto ty1 = UnionType::get(&context, getName("Upad3"));
  ty1.complete(members, /*packed=*/false, /*padding=*/i8,
               RecordType::getAllDataKinds(members));
  auto ty2 = UnionType::get(&context, getName("Upad4"));
  ty2.complete(members, /*packed=*/false, /*padding=*/i8,
               RecordType::getAllDataKinds(members));
  EXPECT_TRUE(ty1.isLayoutIdentical(ty2));
}
