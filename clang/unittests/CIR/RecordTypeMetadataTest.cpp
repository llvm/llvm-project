//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CIR RecordLayoutAttr (module-level ABI metadata).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

class RecordLayoutAttrTest : public ::testing::Test {
protected:
  RecordLayoutAttrTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;

  mlir::StringAttr getName(llvm::StringRef name) {
    return mlir::StringAttr::get(&context, name);
  }
};

TEST_F(RecordLayoutAttrTest, CanPassInRegs) {
  auto attr =
      RecordLayoutAttr::get(&context, ArgPassingKind::CanPassInRegs, true, 4);
  EXPECT_EQ(attr.getArgPassingKind(), ArgPassingKind::CanPassInRegs);
  EXPECT_TRUE(attr.getHasTrivialDtor());
  EXPECT_EQ(attr.getRecordAlign(), 4u);
}

TEST_F(RecordLayoutAttrTest, CannotPassInRegs) {
  auto attr = RecordLayoutAttr::get(&context, ArgPassingKind::CannotPassInRegs,
                                    false, 4);
  EXPECT_EQ(attr.getArgPassingKind(), ArgPassingKind::CannotPassInRegs);
  EXPECT_FALSE(attr.getHasTrivialDtor());
}

TEST_F(RecordLayoutAttrTest, CanNeverPassInRegs) {
  auto attr = RecordLayoutAttr::get(
      &context, ArgPassingKind::CanNeverPassInRegs, false, 8);
  EXPECT_EQ(attr.getArgPassingKind(), ArgPassingKind::CanNeverPassInRegs);
  EXPECT_FALSE(attr.getHasTrivialDtor());
  EXPECT_EQ(attr.getRecordAlign(), 8u);
}

TEST_F(RecordLayoutAttrTest, HighAlignment) {
  auto attr =
      RecordLayoutAttr::get(&context, ArgPassingKind::CanPassInRegs, true, 32);
  EXPECT_EQ(attr.getRecordAlign(), 32u);
}

TEST_F(RecordLayoutAttrTest, RecordTypeUnchanged) {
  IntType i32 = IntType::get(&context, 32, true);
  auto ty =
      RecordType::get(&context, getName("Foo"), RecordType::RecordKind::Struct);
  ty.complete({i32, i32}, /*packed=*/false, /*padded=*/false);
  EXPECT_TRUE(ty.isComplete());
  EXPECT_EQ(ty.getMembers().size(), 2u);
}

TEST_F(RecordLayoutAttrTest, ModuleLevelLookup) {
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(loc);

  auto layoutAttr =
      RecordLayoutAttr::get(&context, ArgPassingKind::CanPassInRegs, true, 8);

  llvm::SmallVector<mlir::NamedAttribute> entries;
  entries.push_back(mlir::NamedAttribute(getName("TestRecord"), layoutAttr));
  module->setAttr(CIRDialect::getRecordLayoutsAttrName(),
                  mlir::DictionaryAttr::get(&context, entries));

  RecordLayoutAttr result = cir::getRecordLayout(module, getName("TestRecord"));
  EXPECT_EQ(result.getArgPassingKind(), ArgPassingKind::CanPassInRegs);
  EXPECT_TRUE(result.getHasTrivialDtor());
  EXPECT_EQ(result.getRecordAlign(), 8u);

  module->erase();
}
