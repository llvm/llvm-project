//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for union RecordType::getTypeSizeInBits.
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
  auto ty = RecordType::get(&context, getName("U"), RecordType::Union);
  ty.complete({i32}, /*packed=*/false, /*padded=*/false);

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
  auto ty = RecordType::get(&context, getName("U2"), RecordType::Union);
  ty.complete({i32, i64}, /*packed=*/false, /*padded=*/false);

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  mlir::DataLayout dl(module);

  llvm::TypeSize size = dl.getTypeSizeInBits(ty);
  EXPECT_EQ(size.getFixedValue(), 64u);

  module->erase();
}

TEST_F(UnionTypeSizeTest, EmptyUnion) {
  auto ty = RecordType::get(&context, getName("Empty"), RecordType::Union);
  ty.complete({}, /*packed=*/false, /*padded=*/false);

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  mlir::DataLayout dl(module);

  llvm::TypeSize size = dl.getTypeSizeInBits(ty);
  EXPECT_EQ(size.getFixedValue(), 0u);

  module->erase();
}
