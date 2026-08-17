//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for per-member record kinds: ABI emptiness and type identity.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

/// Swallows verifier diagnostics, so a getChecked failure can be asserted
/// without the error reaching stderr.
struct ScopedDiagnosticCapture {
  explicit ScopedDiagnosticCapture(MLIRContext &context)
      : handler(&context, [this](mlir::Diagnostic &diag) {
          ++count;
          lastMessage = diag.str();
        }) {}

  unsigned count = 0;
  std::string lastMessage;

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

  RecordType makeStruct(llvm::StringRef name,
                        llvm::ArrayRef<mlir::Type> members,
                        llvm::ArrayRef<RecordMemberKind> kinds) {
    auto ty = StructType::get(&context, getName(name), /*is_class=*/false);
    ty.complete(members, /*packed=*/false, kinds);
    return ty;
  }
};

TEST_F(RecordMemberKindTest, EmptyForTheABIWhenNoMemberHoldsData) {
  IntType u8 = getU8();
  // A record with no members is vacuously empty.
  EXPECT_TRUE(makeStruct("none", {}, {}).isEmptyForABI());
  EXPECT_TRUE(makeStruct("p1", {u8}, {RecordMemberKind::Pad}).isEmptyForABI());
  EXPECT_TRUE(
      makeStruct("e1", {u8}, {RecordMemberKind::Empty}).isEmptyForABI());
  EXPECT_TRUE(makeStruct("pe", {u8, u8},
                         {RecordMemberKind::Pad, RecordMemberKind::Empty})
                  .isEmptyForABI());
  EXPECT_FALSE(
      makeStruct("d1", {u8}, {RecordMemberKind::Data}).isEmptyForABI());
  EXPECT_FALSE(makeStruct("dp", {u8, u8},
                          {RecordMemberKind::Data, RecordMemberKind::Pad})
                   .isEmptyForABI());
}

TEST_F(RecordMemberKindTest, PaddedFollowsThePadKinds) {
  IntType u8 = getU8();
  EXPECT_FALSE(makeStruct("d", {u8}, {RecordMemberKind::Data}).getPadded());
  EXPECT_FALSE(makeStruct("e", {u8}, {RecordMemberKind::Empty}).getPadded());
  EXPECT_TRUE(makeStruct("p", {u8}, {RecordMemberKind::Pad}).getPadded());
  // Interior padding counts too, not just a trailing run.
  EXPECT_TRUE(makeStruct("dpd", {u8, u8, u8},
                         {RecordMemberKind::Data, RecordMemberKind::Pad,
                          RecordMemberKind::Data})
                  .getPadded());
  // An incomplete struct has no members to read a kind from.
  EXPECT_FALSE(StructType::get(&context, getName("inc"), /*is_class=*/false)
                   .getPadded());
}

TEST_F(RecordMemberKindTest, AUnionsPaddingComesFromItsPaddingSlot) {
  IntType u8 = getU8();
  llvm::SmallVector<mlir::Type> members{u8};
  llvm::SmallVector<RecordMemberKind> empty{RecordMemberKind::Empty};
  llvm::ArrayRef<mlir::Type> membersRef(members);

  EXPECT_FALSE(UnionType::get(&context, membersRef, getName("ub"),
                              /*packed=*/false, /*padding=*/mlir::Type{},
                              RecordType::getAllDataKinds(membersRef))
                   .getPadded());
  EXPECT_TRUE(UnionType::get(&context, membersRef, getName("up"),
                             /*packed=*/false, /*padding=*/u8,
                             RecordType::getAllDataKinds(membersRef))
                  .getPadded());
  // An empty kind on a member is not padding.
  EXPECT_FALSE(UnionType::get(&context, membersRef, getName("ue"),
                              /*packed=*/false, /*padding=*/mlir::Type{},
                              llvm::ArrayRef<RecordMemberKind>(empty))
                   .getPadded());
}

TEST_F(RecordMemberKindTest, RejectsAKindListThatDoesNotNameEveryMember) {
  // The assembly syntax cannot express either of these, since it builds one
  // kind per member, but a C++ caller can.
  llvm::SmallVector<mlir::Type> members{getU8(), getU8()};
  llvm::SmallVector<RecordMemberKind> tooFew{RecordMemberKind::Pad};

  ScopedDiagnosticCapture diags(context);
  llvm::ArrayRef<mlir::Type> membersRef(members);
  llvm::ArrayRef<RecordMemberKind> kindsRef(tooFew);
  EXPECT_FALSE(StructType::getChecked(getLoc(), &context, membersRef,
                                      /*packed=*/false, /*is_class=*/false,
                                      kindsRef));
  EXPECT_EQ(diags.count, 1u);
  EXPECT_EQ(diags.lastMessage, "expected 2 member kinds, got 1");

  // An omitted list is not shorthand for all-data.
  EXPECT_FALSE(StructType::getChecked(getLoc(), &context, membersRef,
                                      /*packed=*/false, /*is_class=*/false,
                                      llvm::ArrayRef<RecordMemberKind>{}));
  EXPECT_EQ(diags.count, 2u);
  EXPECT_EQ(diags.lastMessage, "expected 2 member kinds, got 0");

  // A union answers to the same check.
  EXPECT_FALSE(UnionType::getChecked(getLoc(), &context, membersRef,
                                     /*packed=*/false, /*padding=*/mlir::Type{},
                                     llvm::ArrayRef<RecordMemberKind>{}));
  EXPECT_EQ(diags.count, 3u);
  EXPECT_EQ(diags.lastMessage, "expected 2 member kinds, got 0");
}

TEST_F(RecordMemberKindTest, RejectsPadOnAUnionMember) {
  llvm::SmallVector<mlir::Type> members{getU8()};
  llvm::SmallVector<RecordMemberKind> pad{RecordMemberKind::Pad};
  llvm::SmallVector<RecordMemberKind> empty{RecordMemberKind::Empty};

  ScopedDiagnosticCapture diags(context);
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
  RecordType ty = StructType::get(&context, getName("I"), /*is_class=*/false);
  EXPECT_FALSE(ty.isEmptyForABI());
}

TEST_F(RecordMemberKindTest, AUnionsTailPaddingSlotIsNotAMember) {
  IntType u8 = getU8();
  llvm::SmallVector<mlir::Type> members{u8};
  llvm::SmallVector<RecordMemberKind> empty{RecordMemberKind::Empty};
  llvm::ArrayRef<mlir::Type> membersRef(members);

  RecordType allEmpty =
      UnionType::get(&context, membersRef, getName("ue"), /*packed=*/false,
                     /*padding=*/u8, llvm::ArrayRef<RecordMemberKind>(empty));
  EXPECT_TRUE(allEmpty.isEmptyForABI());
  RecordType holdsData =
      UnionType::get(&context, membersRef, getName("ud"), /*packed=*/false,
                     /*padding=*/u8, RecordType::getAllDataKinds(membersRef));
  EXPECT_FALSE(holdsData.isEmptyForABI());
}

TEST_F(RecordMemberKindTest, KindsTakePartInAnonymousTypeIdentity) {
  IntType u8 = getU8();
  auto kindsPad =
      StructType::get(&context, {u8, u8}, /*packed=*/false, /*is_class=*/false,
                      {RecordMemberKind::Data, RecordMemberKind::Pad});
  auto kindsEmpty =
      StructType::get(&context, {u8, u8}, /*packed=*/false, /*is_class=*/false,
                      {RecordMemberKind::Data, RecordMemberKind::Empty});
  EXPECT_NE(kindsPad, kindsEmpty);

  // Kinds are provenance rather than layout.
  EXPECT_TRUE(kindsPad.isLayoutIdentical(kindsEmpty));

  llvm::SmallVector<mlir::Type> unionMembers{u8, u8};
  llvm::SmallVector<RecordMemberKind> unionEmpty{RecordMemberKind::Data,
                                                 RecordMemberKind::Empty};
  llvm::ArrayRef<mlir::Type> unionMembersRef(unionMembers);
  auto unionMarked = UnionType::get(
      &context, unionMembersRef, /*packed=*/false, /*padding=*/mlir::Type{},
      llvm::ArrayRef<RecordMemberKind>(unionEmpty));
  auto unionAllData = UnionType::get(
      &context, unionMembersRef, /*packed=*/false,
      /*padding=*/mlir::Type{}, RecordType::getAllDataKinds(unionMembersRef));
  EXPECT_NE(unionMarked, unionAllData);
  EXPECT_TRUE(unionMarked.isLayoutIdentical(unionAllData));
}
