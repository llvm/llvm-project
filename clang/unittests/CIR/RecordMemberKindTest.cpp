//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for per-member record kinds: what they imply about a record's
// emptiness for the ABI, and how they take part in type identity.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

/// Swallows verifier diagnostics and counts them, so a getChecked failure can
/// be asserted without the error reaching stderr.
struct ScopedDiagnosticCounter {
  explicit ScopedDiagnosticCounter(MLIRContext &context)
      : handler(&context, [this](mlir::Diagnostic &) { ++count; }) {}

  unsigned count = 0;

private:
  mlir::ScopedDiagnosticHandler handler;
};

class RecordMemberKindTest : public ::testing::Test {
protected:
  RecordMemberKindTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;

  mlir::Location getLoc() { return mlir::UnknownLoc::get(&context); }

  mlir::StringAttr getName(llvm::StringRef name) {
    return mlir::StringAttr::get(&context, name);
  }

  IntType getU8() { return IntType::get(&context, 8, false); }

  StructType makeStruct(llvm::StringRef name,
                        llvm::ArrayRef<mlir::Type> members,
                        llvm::ArrayRef<RecordMemberKind> kinds) {
    auto ty = StructType::get(&context, getName(name), /*is_class=*/false);
    ty.complete(members, /*packed=*/false, /*isPadded=*/false, kinds);
    return ty;
  }
};

TEST_F(RecordMemberKindTest, EmptyForTheABIWhenNoMemberHoldsData) {
  IntType u8 = getU8();
  // A record with no members is vacuously empty.
  EXPECT_TRUE(allMembersNonData(makeStruct("none", {}, {})));
  EXPECT_TRUE(
      allMembersNonData(makeStruct("p1", {u8}, {RecordMemberKind::Pad})));
  EXPECT_TRUE(
      allMembersNonData(makeStruct("e1", {u8}, {RecordMemberKind::Empty})));
  EXPECT_TRUE(allMembersNonData(makeStruct(
      "pe", {u8, u8}, {RecordMemberKind::Pad, RecordMemberKind::Empty})));
  // An all-data list is dropped on completion rather than stored, which is the
  // mutate-path half of the canonicalization.
  EXPECT_TRUE(makeStruct("d1", {u8}, {RecordMemberKind::Data})
                  .getMemberKinds()
                  .empty());
  EXPECT_FALSE(allMembersNonData(makeStruct(
      "dp", {u8, u8}, {RecordMemberKind::Data, RecordMemberKind::Pad})));
  // A record with members and no mark list holds data in all of them.
  EXPECT_FALSE(allMembersNonData(makeStruct("unmarked", {u8}, {})));
}

TEST_F(RecordMemberKindTest, RejectsAMarkListThatDoesNotCoverEveryMember) {
  // The assembly syntax cannot express this, since it builds one kind per
  // member, but a C++ caller can.
  llvm::SmallVector<mlir::Type> members{getU8(), getU8()};
  llvm::SmallVector<RecordMemberKind> tooFew{RecordMemberKind::Pad};

  ScopedDiagnosticCounter diags(context);
  llvm::ArrayRef<mlir::Type> membersRef(members);
  llvm::ArrayRef<RecordMemberKind> kindsRef(tooFew);
  EXPECT_FALSE(StructType::getChecked(getLoc(), &context, membersRef,
                                      /*packed=*/false, /*padded=*/false,
                                      /*is_class=*/false, kindsRef));
  EXPECT_EQ(diags.count, 1u);
}

TEST_F(RecordMemberKindTest, RejectsPadOnAUnionMember) {
  // A union's variants all start at offset zero, so there is no inter-member
  // padding a pad mark could describe.
  llvm::SmallVector<mlir::Type> members{getU8()};
  llvm::SmallVector<RecordMemberKind> pad{RecordMemberKind::Pad};
  llvm::SmallVector<RecordMemberKind> empty{RecordMemberKind::Empty};

  ScopedDiagnosticCounter diags(context);
  llvm::ArrayRef<mlir::Type> membersRef(members);
  EXPECT_FALSE(UnionType::getChecked(getLoc(), &context, membersRef,
                                     /*packed=*/false, /*padding=*/mlir::Type{},
                                     llvm::ArrayRef<RecordMemberKind>(pad)));
  EXPECT_EQ(diags.count, 1u);
  EXPECT_TRUE(UnionType::getChecked(getLoc(), &context, membersRef,
                                    /*packed=*/false, /*padding=*/mlir::Type{},
                                    llvm::ArrayRef<RecordMemberKind>(empty)));
  EXPECT_EQ(diags.count, 1u);
}

TEST_F(RecordMemberKindTest, AnIncompleteRecordIsNotEmptyForTheABI) {
  // An incomplete record has no members, which must not read as vacuously
  // holding no data.
  auto ty = StructType::get(&context, getName("I"), /*is_class=*/false);
  EXPECT_FALSE(allMembersNonData(ty));
}

TEST_F(RecordMemberKindTest, AUnionsTailPaddingSlotIsNotAMember) {
  IntType u8 = getU8();
  llvm::SmallVector<mlir::Type> members{u8};
  llvm::SmallVector<RecordMemberKind> empty{RecordMemberKind::Empty};
  llvm::ArrayRef<mlir::Type> membersRef(members);

  auto allEmpty =
      UnionType::get(&context, membersRef, getName("ue"), /*packed=*/false,
                     /*padding=*/u8, llvm::ArrayRef<RecordMemberKind>(empty));
  EXPECT_TRUE(allMembersNonData(allEmpty));
  auto holdsData = UnionType::get(&context, membersRef, getName("ud"),
                                  /*packed=*/false, /*padding=*/u8);
  EXPECT_FALSE(allMembersNonData(holdsData));
}

TEST_F(RecordMemberKindTest, MarksTakePartInAnonymousTypeIdentity) {
  IntType u8 = getU8();
  auto marksPad = StructType::get(
      &context, {u8, u8}, /*packed=*/false, /*padded=*/false,
      /*is_class=*/false, {RecordMemberKind::Data, RecordMemberKind::Pad});
  auto marksEmpty = StructType::get(
      &context, {u8, u8}, /*packed=*/false, /*padded=*/false,
      /*is_class=*/false, {RecordMemberKind::Data, RecordMemberKind::Empty});
  EXPECT_NE(marksPad, marksEmpty);

  // Marks are provenance rather than layout.
  EXPECT_TRUE(marksPad.isLayoutIdentical(marksEmpty));

  llvm::SmallVector<mlir::Type> unionMembers{u8, u8};
  llvm::SmallVector<RecordMemberKind> unionEmpty{RecordMemberKind::Data,
                                                 RecordMemberKind::Empty};
  llvm::ArrayRef<mlir::Type> unionMembersRef(unionMembers);
  auto unionMarked = UnionType::get(
      &context, unionMembersRef, /*packed=*/false, /*padding=*/mlir::Type{},
      llvm::ArrayRef<RecordMemberKind>(unionEmpty));
  auto unionPlain = UnionType::get(&context, unionMembersRef, /*packed=*/false);
  EXPECT_NE(unionMarked, unionPlain);
  EXPECT_TRUE(unionMarked.isLayoutIdentical(unionPlain));
}

TEST_F(RecordMemberKindTest, AnAllDataMarkListIsDropped) {
  IntType u8 = getU8();
  auto allData = StructType::get(
      &context, {u8, u8}, /*packed=*/false, /*padded=*/false,
      /*is_class=*/false, {RecordMemberKind::Data, RecordMemberKind::Data});
  auto noList = StructType::get(&context, {u8, u8}, /*packed=*/false,
                                /*padded=*/false, /*is_class=*/false);
  EXPECT_EQ(allData, noList);
  EXPECT_TRUE(allData.getMemberKinds().empty());
}
