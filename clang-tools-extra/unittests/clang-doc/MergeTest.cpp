//===-- clang-doc/MergeTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocTest.h"
#include "Representation.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

class MergeTest : public ClangDocContextTest {};

TEST_F(MergeTest, mergeNamespaceInfos) {
  NamespaceInfo One;
  One.Name = "Namespace";
  Reference Ns1[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  One.Namespace = llvm::ArrayRef(Ns1);

  Reference RA(NonEmptySID, "ChildNamespace", InfoType::IT_namespace);
  One.Children.Namespaces.push_back(RA);
  Reference RC1(NonEmptySID, "ChildStruct", InfoType::IT_record);
  One.Children.Records.push_back(RC1);

  FunctionInfo F1;
  F1.Name = "OneFunction";
  F1.USR = NonEmptySID;
  One.Children.Functions.push_back(F1);

  EnumInfo E1;
  E1.Name = "OneEnum";
  E1.USR = NonEmptySID;
  One.Children.Enums.push_back(E1);

  NamespaceInfo Two;
  Two.Name = "Namespace";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Reference RB(EmptySID, "OtherChildNamespace", InfoType::IT_namespace);
  Two.Children.Namespaces.push_back(RB);
  Reference RC2(EmptySID, "OtherChildStruct", InfoType::IT_record);
  Two.Children.Records.push_back(RC2);

  FunctionInfo F2;
  F2.Name = "TwoFunction";
  Two.Children.Functions.push_back(F2);

  EnumInfo E2;
  E2.Name = "TwoEnum";
  Two.Children.Enums.push_back(E2);

  OwningPtrVec<Info> Infos;
  Infos.push_back(&One);
  Infos.push_back(&Two);

  NamespaceInfo Expected;
  Expected.Name = "Namespace";
  Reference NsExpected[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsExpected);

  Reference RC(NonEmptySID, "ChildNamespace", InfoType::IT_namespace);
  Expected.Children.Namespaces.push_back(RC);
  Reference RCE1(NonEmptySID, "ChildStruct", InfoType::IT_record);
  Expected.Children.Records.push_back(RCE1);
  Reference RD(EmptySID, "OtherChildNamespace", InfoType::IT_namespace);
  Expected.Children.Namespaces.push_back(RD);
  Reference RCE2(EmptySID, "OtherChildStruct", InfoType::IT_record);
  Expected.Children.Records.push_back(RCE2);

  FunctionInfo FE1;
  FE1.Name = "OneFunction";
  FE1.USR = NonEmptySID;
  Expected.Children.Functions.push_back(FE1);

  FunctionInfo FE2;
  FE2.Name = "TwoFunction";
  Expected.Children.Functions.push_back(FE2);

  EnumInfo EE1;
  EE1.Name = "OneEnum";
  EE1.USR = NonEmptySID;
  Expected.Children.Enums.push_back(EE1);

  EnumInfo EE2;
  EE2.Name = "TwoEnum";
  Expected.Children.Enums.push_back(EE2);

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckNamespaceInfo(InfoAsNamespace(&Expected), InfoAsNamespace(Actual.get()));
}

TEST_F(MergeTest, mergeRecordInfos) {
  RecordInfo One;
  One.Name = "r";
  One.IsTypeDef = true;
  Reference Ns1[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  One.Namespace = llvm::ArrayRef(Ns1);

  One.DefLoc = Location(10, 10, "test.cpp");

  MemberTypeInfo M1[] = {
      MemberTypeInfo(TypeInfo("int"), "X", AccessSpecifier::AS_private)};
  One.Members = llvm::ArrayRef(M1);
  One.TagType = TagTypeKind::Class;
  Reference P1[] = {Reference(EmptySID, "F", InfoType::IT_record)};
  One.Parents = llvm::ArrayRef(P1);
  Reference VP1[] = {Reference(EmptySID, "G", InfoType::IT_record)};
  One.VirtualParents = llvm::ArrayRef(VP1);

  BaseRecordInfo B1[] = {BaseRecordInfo(EmptySID, "F", "path/to/F", true,
                                        AccessSpecifier::AS_protected, true)};
  One.Bases = llvm::ArrayRef(B1);
  Reference RCShared1(NonEmptySID, "SharedChildStruct", InfoType::IT_record);
  One.Children.Records.push_back(RCShared1);

  FunctionInfo F1;
  F1.Name = "OneFunction";
  F1.USR = NonEmptySID;
  One.Children.Functions.push_back(F1);

  EnumInfo E1;
  E1.Name = "OneEnum";
  E1.USR = NonEmptySID;
  One.Children.Enums.push_back(E1);

  RecordInfo Two;
  Two.Name = "r";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Location Loc2(12, 12, "test.cpp");
  Two.Loc.push_back(Loc2);

  Two.TagType = TagTypeKind::Class;

  Reference RCShared2(NonEmptySID, "SharedChildStruct", InfoType::IT_record,
                      "path");
  Two.Children.Records.push_back(RCShared2);

  FunctionInfo F2;
  F2.Name = "TwoFunction";
  Two.Children.Functions.push_back(F2);

  EnumInfo E2;
  E2.Name = "TwoEnum";
  Two.Children.Enums.push_back(E2);

  OwningPtrVec<Info> Infos;
  Infos.push_back(&One);
  Infos.push_back(&Two);

  RecordInfo Expected;
  Expected.Name = "r";
  Expected.IsTypeDef = true;
  Reference NsE[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsE);

  Expected.DefLoc = Location(10, 10, "test.cpp");
  Location LocE(12, 12, "test.cpp");
  Expected.Loc.push_back(LocE);

  MemberTypeInfo ME[] = {
      MemberTypeInfo(TypeInfo("int"), "X", AccessSpecifier::AS_private)};
  Expected.Members = llvm::ArrayRef(ME);
  Expected.TagType = TagTypeKind::Class;
  Reference PE[] = {Reference(EmptySID, "F", InfoType::IT_record)};
  Expected.Parents = llvm::ArrayRef(PE);
  Reference VPE[] = {Reference(EmptySID, "G", InfoType::IT_record)};
  Expected.VirtualParents = llvm::ArrayRef(VPE);
  BaseRecordInfo BE[] = {BaseRecordInfo(EmptySID, "F", "path/to/F", true,
                                        AccessSpecifier::AS_protected, true)};
  Expected.Bases = llvm::ArrayRef(BE);

  Reference RCSharedE(NonEmptySID, "SharedChildStruct", InfoType::IT_record,
                      "path");
  Expected.Children.Records.push_back(RCSharedE);
  FunctionInfo FE1;
  FE1.Name = "OneFunction";
  FE1.USR = NonEmptySID;
  Expected.Children.Functions.push_back(FE1);

  FunctionInfo FE2;
  FE2.Name = "TwoFunction";
  Expected.Children.Functions.push_back(FE2);

  EnumInfo EE1;
  EE1.Name = "OneEnum";
  EE1.USR = NonEmptySID;
  Expected.Children.Enums.push_back(EE1);

  EnumInfo EE2;
  EE2.Name = "TwoEnum";
  Expected.Children.Enums.push_back(EE2);

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckRecordInfo(InfoAsRecord(&Expected), InfoAsRecord(Actual.get()));
}

TEST_F(MergeTest, mergeFunctionInfos) {
  FunctionInfo One;
  One.Name = "f";
  Reference Ns1[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  One.Namespace = llvm::ArrayRef(Ns1);

  One.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  One.Loc.push_back(Loc1);

  One.IsMethod = true;
  One.Parent = Reference(EmptySID, "Parent", InfoType::IT_namespace);

  CommentInfo OneText[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, "This is a text comment.")};
  CommentInfo OnePara[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, OneText)};
  CommentInfo TopOne(CommentKind::CK_FullComment, OnePara);
  One.Description.push_back(TopOne);

  FunctionInfo Two;
  Two.Name = "f";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Location Loc2(12, 12, "test.cpp");
  Two.Loc.push_back(Loc2);

  Two.ReturnType = TypeInfo("void");
  FieldTypeInfo P2(TypeInfo("int"), "P");
  FieldTypeInfo Params2[] = {std::move(P2)};
  Two.Params = llvm::ArrayRef(Params2);

  CommentInfo TwoText[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, "This is a text comment.")};
  CommentInfo TwoPara[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, TwoText)};
  CommentInfo TopTwo(CommentKind::CK_FullComment, TwoPara);
  Two.Description.push_back(TopTwo);

  OwningPtrVec<Info> Infos;
  Infos.push_back(&One);
  Infos.push_back(&Two);

  FunctionInfo Expected;
  Expected.Name = "f";
  Reference NsE[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsE);

  Expected.DefLoc = Location(10, 10, "test.cpp");
  Location LocE(12, 12, "test.cpp");
  Expected.Loc.push_back(LocE);

  Expected.ReturnType = TypeInfo("void");
  FieldTypeInfo PE(TypeInfo("int"), "P");
  FieldTypeInfo ParamsE[] = {std::move(PE)};
  Expected.Params = llvm::ArrayRef(ParamsE);
  Expected.IsMethod = true;
  Expected.Parent = Reference(EmptySID, "Parent", InfoType::IT_namespace);

  CommentInfo ExpectedText[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, "This is a text comment.")};
  CommentInfo ExpectedPara[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, ExpectedText)};
  CommentInfo TopE(CommentKind::CK_FullComment, ExpectedPara);
  Expected.Description.push_back(TopE);

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckFunctionInfo(InfoAsFunction(&Expected), InfoAsFunction(Actual.get()));
}

TEST_F(MergeTest, mergeEnumInfos) {
  EnumInfo One;
  One.Name = "e";
  Reference Ns1[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  One.Namespace = llvm::ArrayRef(Ns1);

  One.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  One.Loc.push_back(Loc1);

  One.Scoped = true;

  EnumInfo Two;
  Two.Name = "e";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Location Loc2(20, 20, "test.cpp");
  Two.Loc.push_back(Loc2);

  EnumValueInfo EV2[] = {EnumValueInfo("X"), EnumValueInfo("Y")};
  Two.Members = llvm::ArrayRef(EV2);

  OwningPtrVec<Info> Infos;
  Infos.push_back(&One);
  Infos.push_back(&Two);

  EnumInfo Expected;
  Expected.Name = "e";
  Reference NsE[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsE);

  Expected.DefLoc = Location(10, 10, "test.cpp");
  Location LocE1(12, 12, "test.cpp");
  Expected.Loc.push_back(LocE1);
  Location LocE2(20, 20, "test.cpp");
  Expected.Loc.push_back(LocE2);

  EnumValueInfo EV_E[] = {EnumValueInfo("X"), EnumValueInfo("Y")};
  Expected.Members = llvm::ArrayRef(EV_E);
  Expected.Scoped = true;

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckEnumInfo(InfoAsEnum(&Expected), InfoAsEnum(Actual.get()));
}

} // namespace doc
} // namespace clang
