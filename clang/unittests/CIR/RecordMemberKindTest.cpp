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

#include "mlir/IR/BuiltinOps.h"
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

  /// One bit-field as the source declared it.
  BitFieldDeclAttr getDecl(mlir::Type declaredType, uint64_t width,
                           bool isUnnamed = false) {
    return BitFieldDeclAttr::get(declaredType, width, isUnnamed);
  }

  /// An access unit stored as \p storageType, holding \p fields.
  BitFieldType getUnit(mlir::Type storageType,
                       llvm::ArrayRef<BitFieldDeclAttr> fields) {
    return BitFieldType::get(&context, storageType, fields);
  }

  /// A zero-width bit-field declared with \p declaredType, which belongs to no
  /// access unit.
  BitFieldType getZeroWidth(mlir::Type declaredType) {
    return BitFieldType::get(&context, /*storage_type=*/mlir::Type{},
                             getDecl(declaredType, 0, /*isUnnamed=*/true));
  }

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
  // A named bit-field holds data the same way a field does.
  EXPECT_FALSE(
      makeStruct("b1", {u8}, {RecordMemberKind::BitField}).isEmptyForABI());
  EXPECT_FALSE(makeStruct("be", {u8, u8},
                          {RecordMemberKind::BitField, RecordMemberKind::Empty})
                   .isEmptyForABI());
}

TEST_F(RecordMemberKindTest, AnAccessUnitHoldsTheFieldsTheSourceDeclared) {
  IntType s32 = IntType::get(&context, 32, true);
  IntType s64 = IntType::get(&context, 64, true);
  IntType u32 = IntType::get(&context, 32, false);

  // struct { int a : 4; long long b : 27; int : 1; };  One unit, three fields,
  // and neither the declared types nor the names survive in the unit itself.
  BitFieldType unit = getUnit(u32, {getDecl(s32, 4), getDecl(s64, 27),
                                    getDecl(s32, 1, /*isUnnamed=*/true)});
  EXPECT_TRUE(unit.ownsBytes());
  EXPECT_FALSE(unit.isZeroWidth());
  EXPECT_TRUE(unit.holdsNamedField());
  EXPECT_TRUE(cir::memberOwnsBytes(unit));
  EXPECT_EQ(cir::memberStorageType(unit), u32);

  EXPECT_EQ(unit.getFields().size(), 3u);
  EXPECT_EQ(unit.getFields()[1].getDeclaredType(), s64);
  EXPECT_EQ(unit.getFields()[1].getWidth(), 27u);

  // The fields are contiguous and in declaration order, so each starts where
  // the ones ahead of it end.
  EXPECT_EQ(unit.getFieldBitOffset(0), 0u);
  EXPECT_EQ(unit.getFieldBitOffset(1), 4u);
  EXPECT_EQ(unit.getFieldBitOffset(2), 31u);

  // A unit no field of the source names is storage all the same.
  BitFieldType unnamed = getUnit(u32, getDecl(s32, 8, /*isUnnamed=*/true));
  EXPECT_FALSE(unnamed.holdsNamedField());
  EXPECT_TRUE(unnamed.ownsBytes());

  // A zero-width bit-field belongs to no unit and has no storage.
  BitFieldType zeroWidth = getZeroWidth(s32);
  EXPECT_FALSE(zeroWidth.ownsBytes());
  EXPECT_TRUE(zeroWidth.isZeroWidth());
  EXPECT_FALSE(cir::memberOwnsBytes(zeroWidth));
  EXPECT_FALSE(zeroWidth.holdsNamedField());

  // Any other member owns its own bytes and is its own storage.
  EXPECT_TRUE(cir::memberOwnsBytes(s32));
  EXPECT_EQ(cir::memberStorageType(s32), s32);
}

TEST_F(RecordMemberKindTest, RejectsAnAccessUnitThatHoldsNothingCoherent) {
  IntType s32 = IntType::get(&context, 32, true);
  IntType u32 = IntType::get(&context, 32, false);

  ScopedDiagnosticCapture diags(context);

  // A unit exists to hold bit-fields.
  EXPECT_FALSE(BitFieldType::getChecked(getLoc(), &context, mlir::Type(u32),
                                        llvm::ArrayRef<BitFieldDeclAttr>{}));
  EXPECT_EQ(diags.count, 1u);
  EXPECT_EQ(diags.lastMessage,
            "bit-field member must hold at least one bit-field");

  // Storage is an access unit, never another bit-field.
  BitFieldType unit = getUnit(u32, getDecl(s32, 4));
  llvm::SmallVector<BitFieldDeclAttr> oneField{getDecl(s32, 4)};
  EXPECT_FALSE(
      BitFieldType::getChecked(getLoc(), &context, mlir::Type(unit),
                               llvm::ArrayRef<BitFieldDeclAttr>(oneField)));
  EXPECT_EQ(diags.count, 2u);
  EXPECT_EQ(diags.lastMessage, "bit-field access unit storage cannot itself "
                               "be a bit-field type");

  // A zero-width bit-field ends the run before it rather than joining a unit.
  llvm::SmallVector<BitFieldDeclAttr> withZeroWidth{
      getDecl(s32, 4), getDecl(s32, 0, /*isUnnamed=*/true)};
  EXPECT_FALSE(BitFieldType::getChecked(
      getLoc(), &context, mlir::Type(u32),
      llvm::ArrayRef<BitFieldDeclAttr>(withZeroWidth)));
  EXPECT_EQ(diags.count, 3u);
  EXPECT_EQ(diags.lastMessage,
            "a zero-width bit-field cannot occupy an access unit");

  // And a member with no storage is that zero-width bit-field, nothing else.
  llvm::SmallVector<BitFieldDeclAttr> holdsBits{
      getDecl(s32, 3, /*isUnnamed=*/true)};
  EXPECT_FALSE(
      BitFieldType::getChecked(getLoc(), &context, mlir::Type{},
                               llvm::ArrayRef<BitFieldDeclAttr>(holdsBits)));
  EXPECT_EQ(diags.count, 4u);
  EXPECT_EQ(diags.lastMessage, "a bit-field member without storage must hold "
                               "a single zero-width bit-field");

  // A unit and a zero-width bit-field after it is what a run looks like.
  llvm::SmallVector<mlir::Type> run{unit, getZeroWidth(s32)};
  llvm::SmallVector<RecordMemberKind> runKinds{RecordMemberKind::BitField,
                                               RecordMemberKind::Empty};
  EXPECT_TRUE(StructType::getChecked(
      getLoc(), &context, llvm::ArrayRef<mlir::Type>(run),
      /*packed=*/false, /*is_class=*/false,
      llvm::ArrayRef<RecordMemberKind>(runKinds)));
  EXPECT_EQ(diags.count, 4u);
}

TEST_F(RecordMemberKindTest, OnlyAZeroWidthBitFieldHoldsNoDataForTheABI) {
  IntType u8 = getU8();
  IntType s32 = IntType::get(&context, 32, true);
  IntType u32 = IntType::get(&context, 32, false);
  BitFieldType zeroWidth = getZeroWidth(s32);
  auto zeroLen = cir::ArrayType::get(s32, 0);

  // The mark tells a unit that some field of the source names from one that
  // none does, and only the latter can be empty for the ABI.
  EXPECT_TRUE(cir::holdsDataForABI(RecordMemberKind::BitField));
  EXPECT_TRUE(cir::holdsDataForABI(RecordMemberKind::Data));
  EXPECT_FALSE(cir::holdsDataForABI(RecordMemberKind::Pad));
  EXPECT_FALSE(cir::holdsDataForABI(RecordMemberKind::Empty));

  // A record of nothing but zero-width bit-fields declares no storage.
  EXPECT_TRUE(
      makeStruct("zw", {zeroWidth}, {RecordMemberKind::Empty}).isEmptyForABI());
  EXPECT_TRUE(makeStruct("zwzw", {zeroWidth, zeroWidth},
                         {RecordMemberKind::Empty, RecordMemberKind::Empty})
                  .isEmptyForABI());
  EXPECT_TRUE(makeStruct("zwpad", {zeroWidth, u8},
                         {RecordMemberKind::Empty, RecordMemberKind::Pad})
                  .isEmptyForABI());
  // A unit of unnamed bit-fields is storage the classifier reads, so the mark
  // alone does not answer this: whether the unit holds bytes does.
  EXPECT_FALSE(makeStruct("unnamed",
                          {getUnit(u32, getDecl(s32, 8, /*isUnnamed=*/true))},
                          {RecordMemberKind::Empty})
                   .isEmptyForABI());
  EXPECT_FALSE(makeStruct("unnamedRun",
                          {getUnit(u32, {getDecl(s32, 8, /*isUnnamed=*/true),
                                         getDecl(s32, 4, /*isUnnamed=*/true)})},
                          {RecordMemberKind::Empty})
                   .isEmptyForABI());
  // A zero-length array under `data` is a flexible array member, which holds
  // data.
  EXPECT_FALSE(
      makeStruct("fam", {zeroLen}, {RecordMemberKind::Data}).isEmptyForABI());
  EXPECT_FALSE(makeStruct("zwdata", {zeroWidth, u8},
                          {RecordMemberKind::Empty, RecordMemberKind::Data})
                   .isEmptyForABI());
}

TEST_F(RecordMemberKindTest, AZeroWidthBitFieldLendsNoSizeOrAlignment) {
  IntType s8 = IntType::get(&context, 8, true);
  IntType s64 = IntType::get(&context, 64, true);
  IntType u8 = getU8();
  BitFieldType zeroWidth = getZeroWidth(s64);
  auto pad7 = cir::ArrayType::get(u8, 7);

  // Without the member the record is two bytes at alignment one.  The declared
  // type would take it to sixteen bytes at alignment eight.
  mlir::Type members[] = {s8, pad7, zeroWidth, s8};
  cir::RecordMemberKind kinds[] = {
      RecordMemberKind::Data, RecordMemberKind::Pad, RecordMemberKind::Empty,
      RecordMemberKind::Data};
  auto ty = StructType::get(&context, getName("ZwLayout"), /*is_class=*/false);
  ty.complete(members, /*packed=*/false, kinds);

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  mlir::DataLayout dl(module);

  EXPECT_EQ(dl.getTypeSizeInBits(ty).getFixedValue(), 72u);
  EXPECT_EQ(dl.getTypeABIAlignment(ty), 1u);
  // The member sits where the storage ahead of it ends, and the member after it
  // is not pushed along by the declared type's alignment.
  EXPECT_EQ(ty.getElementOffset(dl, 2), 8u);
  EXPECT_EQ(ty.getElementOffset(dl, 3), 8u);

  module->erase();
}

TEST_F(RecordMemberKindTest, ATrailingZeroWidthBitFieldLeavesTailPadding) {
  IntType s8 = IntType::get(&context, 8, true);
  IntType s32 = IntType::get(&context, 32, true);
  IntType u8 = getU8();
  BitFieldType zeroWidth = getZeroWidth(s32);
  auto pad3 = cir::ArrayType::get(u8, 3);

  // The trailing run is the pad plus the zero-width bit-field, so a derived
  // class may reuse every byte after the first.
  mlir::Type members[] = {s8, pad3, zeroWidth};
  cir::RecordMemberKind kinds[] = {
      RecordMemberKind::Data, RecordMemberKind::Pad, RecordMemberKind::Empty};
  auto ty = StructType::get(&context, getName("ZwTail"), /*is_class=*/false);
  ty.complete(members, /*packed=*/false, kinds);

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  mlir::DataLayout dl(module);

  EXPECT_EQ(ty.computeStructDataSize(dl), 1u);

  module->erase();
}

TEST_F(RecordMemberKindTest, ARunOfBitFieldsIsOneMemberOfTheRecord) {
  IntType s32 = IntType::get(&context, 32, true);
  IntType u16 = IntType::get(&context, 16, false);
  IntType u8 = getU8();
  auto pad2 = cir::ArrayType::get(u8, 2);

  // struct { int a; int b : 3; int : 5; int c : 8; int : 0; };  The three
  // bit-fields share one two-byte unit, the zero-width one holds no storage,
  // and the pad fills the record out to eight bytes.
  mlir::Type members[] = {
      s32,
      getUnit(u16, {getDecl(s32, 3), getDecl(s32, 5, true), getDecl(s32, 8)}),
      getZeroWidth(s32), pad2};
  cir::RecordMemberKind kinds[] = {
      RecordMemberKind::Data, RecordMemberKind::BitField,
      RecordMemberKind::Empty, RecordMemberKind::Pad};
  auto ty = StructType::get(&context, getName("Run"), /*is_class=*/false);
  ty.complete(members, /*packed=*/false, kinds);

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  mlir::DataLayout dl(module);

  // The unit lends the record the size of its storage, and the zero-width
  // bit-field lends it nothing.
  EXPECT_EQ(dl.getTypeSizeInBits(ty).getFixedValue(), 64u);
  EXPECT_EQ(dl.getTypeABIAlignment(ty), 4u);

  EXPECT_EQ(ty.getElementOffset(dl, 1), 4u);
  EXPECT_EQ(ty.getElementOffset(dl, 2), 6u);
  EXPECT_EQ(ty.getElementOffset(dl, 3), 6u);

  // The lowered body drops the member that holds no storage, so the members
  // behind it shift down.
  EXPECT_EQ(ty.getLLVMFieldIndex(0), 0u);
  EXPECT_EQ(ty.getLLVMFieldIndex(1), 1u);
  EXPECT_EQ(ty.getLLVMFieldIndex(3), 2u);

  module->erase();
}

TEST_F(RecordMemberKindTest, PaddedFollowsThePadKinds) {
  IntType u8 = getU8();
  EXPECT_FALSE(makeStruct("d", {u8}, {RecordMemberKind::Data}).getPadded());
  EXPECT_FALSE(makeStruct("e", {u8}, {RecordMemberKind::Empty}).getPadded());
  // A bit-field's storage is declared, so it is not padding.
  EXPECT_FALSE(makeStruct("b", {u8}, {RecordMemberKind::BitField}).getPadded());
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
  // A union variant can be a bit-field, which is not padding.
  llvm::SmallVector<RecordMemberKind> bitField{RecordMemberKind::BitField};
  EXPECT_TRUE(
      UnionType::getChecked(getLoc(), &context, membersRef,
                            /*packed=*/false, /*padding=*/mlir::Type{},
                            llvm::ArrayRef<RecordMemberKind>(bitField)));
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

  auto kindsBitField =
      StructType::get(&context, {u8, u8}, /*packed=*/false, /*is_class=*/false,
                      {RecordMemberKind::BitField, RecordMemberKind::Pad});
  EXPECT_NE(kindsPad, kindsBitField);
  EXPECT_TRUE(kindsPad.isLayoutIdentical(kindsBitField));

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
