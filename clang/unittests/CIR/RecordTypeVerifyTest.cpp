//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests that every way of building a record runs its verifier: get(),
// sub-element replacement, and completion.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

namespace {

class RecordTypeVerifyTest : public ::testing::Test {
protected:
  RecordTypeVerifyTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;

  StringAttr getName(llvm::StringRef name) {
    return StringAttr::get(&context, name);
  }

  IntType getU8() { return IntType::get(&context, 8, false); }

  /// A zero-width bit-field, which owns no storage.
  BitFieldType getZeroWidth() {
    IntType s32 = IntType::get(&context, 32, true);
    return BitFieldType::get(
        &context, /*storage_type=*/Type{},
        BitFieldDeclAttr::get(s32, /*width=*/0, /*isUnnamed=*/true));
  }
};

TEST_F(RecordTypeVerifyTest, ReplacingWithValidMembersSucceeds) {
  IntType u8 = getU8();
  IntType u16 = IntType::get(&context, 16, false);
  auto anon = StructType::get(&context, {u8}, /*packed=*/false,
                              /*is_class=*/false, {RecordMemberKind::Data});

  AttrTypeReplacer replacer;
  replacer.addReplacement([&](IntType ty) -> std::optional<Type> {
    if (ty == u8)
      return u16;
    return std::nullopt;
  });
  auto replaced = mlir::dyn_cast<StructType>(replacer.replace(anon));
  ASSERT_TRUE(replaced);
  EXPECT_EQ(replaced.getMembers().front(), u16);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)

TEST_F(RecordTypeVerifyTest, StructGetVerifies) {
  EXPECT_DEATH(StructType::get(&context, {getU8()}, getName(""),
                               /*packed=*/false, /*is_class=*/false,
                               {RecordMemberKind::Data}),
               "identified records cannot have an empty name");
}

TEST_F(RecordTypeVerifyTest, UnionGetVerifies) {
  EXPECT_DEATH(UnionType::get(&context, {getU8()}, getName("U"),
                              /*packed=*/false, /*padding=*/Type{},
                              {RecordMemberKind::Pad}),
               "a union member cannot be marked pad");
}

TEST_F(RecordTypeVerifyTest, ReplacingAnonymousRecordMembersVerifies) {
  IntType u8 = getU8();
  auto anon = UnionType::get(&context, {u8}, /*packed=*/false,
                             /*padding=*/Type{}, {RecordMemberKind::Data});

  AttrTypeReplacer replacer;
  replacer.addReplacement([&](IntType ty) -> std::optional<Type> {
    if (ty == u8)
      return getZeroWidth();
    return std::nullopt;
  });
  EXPECT_DEATH(replacer.replace(anon), "must own its access unit storage");
}

TEST_F(RecordTypeVerifyTest, StructCompletionRejectsForeignMember) {
  auto ty = StructType::get(&context, getName("S"), /*is_class=*/false);
  EXPECT_DEATH(ty.complete({IntegerType::get(&context, 32)}, /*packed=*/false,
                           {RecordMemberKind::Data}),
               "record member at index 0 must be a CIR type");
}

TEST_F(RecordTypeVerifyTest, UnionCompletionRejectsForeignPadding) {
  auto ty = UnionType::get(&context, getName("U"));
  EXPECT_DEATH(ty.complete({getU8()}, /*packed=*/false,
                           /*padding=*/IntegerType::get(&context, 8),
                           {RecordMemberKind::Data}),
               "union padding must be !u8i");
}

#endif

} // namespace
