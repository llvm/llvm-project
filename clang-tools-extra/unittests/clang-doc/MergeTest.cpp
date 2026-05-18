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
  InfoNode<Reference> RANode(&RA);
  One.Children.Namespaces.push_back(RANode);
  Reference RC1(NonEmptySID, "ChildStruct", InfoType::IT_record);
  InfoNode<Reference> RC1Node(&RC1);
  One.Children.Records.push_back(RC1Node);

  FunctionInfo F1;
  F1.Name = "OneFunction";
  F1.USR = NonEmptySID;
  InfoNode<FunctionInfo> F1Node(&F1);
  One.Children.Functions.push_back(F1Node);

  EnumInfo E1;
  E1.Name = "OneEnum";
  E1.USR = NonEmptySID;
  InfoNode<EnumInfo> E1Node(&E1);
  One.Children.Enums.push_back(E1Node);

  NamespaceInfo Two;
  Two.Name = "Namespace";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Reference RB(EmptySID, "OtherChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RBNode(&RB);
  Two.Children.Namespaces.push_back(RBNode);
  Reference RC2(EmptySID, "OtherChildStruct", InfoType::IT_record);
  InfoNode<Reference> RC2Node(&RC2);
  Two.Children.Records.push_back(RC2Node);

  FunctionInfo F2;
  F2.Name = "TwoFunction";
  InfoNode<FunctionInfo> F2Node(&F2);
  Two.Children.Functions.push_back(F2Node);

  EnumInfo E2;
  E2.Name = "TwoEnum";
  InfoNode<EnumInfo> E2Node(&E2);
  Two.Children.Enums.push_back(E2Node);

  OwningPtrVec<Info> Infos;
  Infos.push_back(&One);
  Infos.push_back(&Two);

  NamespaceInfo Expected;
  Expected.Name = "Namespace";
  Reference NsExpected[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsExpected);

  Reference RC(NonEmptySID, "ChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RCNode(&RC);
  Expected.Children.Namespaces.push_back(RCNode);
  Reference RCE1(NonEmptySID, "ChildStruct", InfoType::IT_record);
  InfoNode<Reference> RCE1Node(&RCE1);
  Expected.Children.Records.push_back(RCE1Node);
  Reference RD(EmptySID, "OtherChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RDNode(&RD);
  Expected.Children.Namespaces.push_back(RDNode);
  Reference RCE2(EmptySID, "OtherChildStruct", InfoType::IT_record);
  InfoNode<Reference> RCE2Node(&RCE2);
  Expected.Children.Records.push_back(RCE2Node);

  FunctionInfo FE1;
  FE1.Name = "OneFunction";
  FE1.USR = NonEmptySID;
  InfoNode<FunctionInfo> FE1Node(&FE1);
  Expected.Children.Functions.push_back(FE1Node);

  FunctionInfo FE2;
  FE2.Name = "TwoFunction";
  InfoNode<FunctionInfo> FE2Node(&FE2);
  Expected.Children.Functions.push_back(FE2Node);

  EnumInfo EE1;
  EE1.Name = "OneEnum";
  EE1.USR = NonEmptySID;
  InfoNode<EnumInfo> EE1Node(&EE1);
  Expected.Children.Enums.push_back(EE1Node);

  EnumInfo EE2;
  EE2.Name = "TwoEnum";
  InfoNode<EnumInfo> EE2Node(&EE2);
  Expected.Children.Enums.push_back(EE2Node);

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckNamespaceInfo(InfoAsNamespace(&Expected), InfoAsNamespace(Actual.get()));
}

TEST_F(MergeTest, mergeSingleNamespaceInfo) {
  NamespaceInfo One;
  One.Name = "Namespace";
  Reference Ns1[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  One.Namespace = llvm::ArrayRef(Ns1);

  Reference RA(NonEmptySID, "ChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RANode(&RA);
  One.Children.Namespaces.push_back(RANode);
  Reference RC1(NonEmptySID, "ChildStruct", InfoType::IT_record);
  InfoNode<Reference> RC1Node(&RC1);
  One.Children.Records.push_back(RC1Node);

  FunctionInfo F1;
  F1.Name = "OneFunction";
  F1.USR = NonEmptySID;
  InfoNode<FunctionInfo> F1Node(&F1);
  One.Children.Functions.push_back(F1Node);

  EnumInfo E1;
  E1.Name = "OneEnum";
  E1.USR = NonEmptySID;
  InfoNode<EnumInfo> E1Node(&E1);
  One.Children.Enums.push_back(E1Node);

  NamespaceInfo Two;
  Two.Name = "Namespace";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Reference RB(EmptySID, "OtherChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RBNode(&RB);
  Two.Children.Namespaces.push_back(RBNode);
  Reference RC2(EmptySID, "OtherChildStruct", InfoType::IT_record);
  InfoNode<Reference> RC2Node(&RC2);
  Two.Children.Records.push_back(RC2Node);

  FunctionInfo F2;
  F2.Name = "TwoFunction";
  InfoNode<FunctionInfo> F2Node(&F2);
  Two.Children.Functions.push_back(F2Node);

  EnumInfo E2;
  E2.Name = "TwoEnum";
  InfoNode<EnumInfo> E2Node(&E2);
  Two.Children.Enums.push_back(E2Node);

  NamespaceInfo Expected;
  Expected.Name = "Namespace";
  Reference NsExpected[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsExpected);

  Reference RC(NonEmptySID, "ChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RCNode(&RC);
  Expected.Children.Namespaces.push_back(RCNode);
  Reference RCE1(NonEmptySID, "ChildStruct", InfoType::IT_record);
  InfoNode<Reference> RCE1Node(&RCE1);
  Expected.Children.Records.push_back(RCE1Node);
  Reference RD(EmptySID, "OtherChildNamespace", InfoType::IT_namespace);
  InfoNode<Reference> RDNode(&RD);
  Expected.Children.Namespaces.push_back(RDNode);
  Reference RCE2(EmptySID, "OtherChildStruct", InfoType::IT_record);
  InfoNode<Reference> RCE2Node(&RCE2);
  Expected.Children.Records.push_back(RCE2Node);

  FunctionInfo FE1;
  FE1.Name = "OneFunction";
  FE1.USR = NonEmptySID;
  InfoNode<FunctionInfo> FE1Node(&FE1);
  Expected.Children.Functions.push_back(FE1Node);

  FunctionInfo FE2;
  FE2.Name = "TwoFunction";
  InfoNode<FunctionInfo> FE2Node(&FE2);
  Expected.Children.Functions.push_back(FE2Node);

  EnumInfo EE1;
  EE1.Name = "OneEnum";
  EE1.USR = NonEmptySID;
  InfoNode<EnumInfo> EE1Node(&EE1);
  Expected.Children.Enums.push_back(EE1Node);

  EnumInfo EE2;
  EE2.Name = "TwoEnum";
  InfoNode<EnumInfo> EE2Node(&EE2);
  Expected.Children.Enums.push_back(EE2Node);
  NamespaceInfo ReducedObj;
  ReducedObj.IT = InfoType::IT_namespace;
  doc::OwnedPtr<doc::Info> Reduced = &ReducedObj;

  Info *PtrOne = &One;
  auto Err1 = mergeSingleInfo(Reduced, std::move(PtrOne), doc::PersistentArena);
  assert(!Err1);

  Info *PtrTwo = &Two;
  auto Err2 = mergeSingleInfo(Reduced, std::move(PtrTwo), doc::PersistentArena);
  assert(!Err2);

  CheckNamespaceInfo(InfoAsNamespace(&Expected),
                     static_cast<NamespaceInfo *>(getPtr(Reduced)));

  auto *RedNS = static_cast<NamespaceInfo *>(getPtr(Reduced));
  // Check that children functions are NOT the same instances as in One or Two
  ASSERT_NE(RedNS->Children.Functions.front().Ptr, &F1);
  ASSERT_NE(RedNS->Children.Functions.back().Ptr, &F2);
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
  InfoNode<Reference> RCShared1Node(&RCShared1);
  One.Children.Records.push_back(RCShared1Node);

  FunctionInfo F1;
  F1.Name = "OneFunction";
  F1.USR = NonEmptySID;
  InfoNode<FunctionInfo> F1Node(&F1);
  One.Children.Functions.push_back(F1Node);

  EnumInfo E1;
  E1.Name = "OneEnum";
  E1.USR = NonEmptySID;
  InfoNode<EnumInfo> E1Node(&E1);
  One.Children.Enums.push_back(E1Node);

  RecordInfo Two;
  Two.Name = "r";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Location Loc2(12, 12, "test.cpp");
  InfoNode<Location> Loc2Node(&Loc2);
  Two.Loc.push_back(Loc2Node);

  Two.TagType = TagTypeKind::Class;

  Reference RCShared2(NonEmptySID, "SharedChildStruct", InfoType::IT_record,
                      "path");
  InfoNode<Reference> RCShared2Node(&RCShared2);
  Two.Children.Records.push_back(RCShared2Node);

  FunctionInfo F2;
  F2.Name = "TwoFunction";
  InfoNode<FunctionInfo> F2Node(&F2);
  Two.Children.Functions.push_back(F2Node);

  EnumInfo E2;
  E2.Name = "TwoEnum";
  InfoNode<EnumInfo> E2Node(&E2);
  Two.Children.Enums.push_back(E2Node);

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
  InfoNode<Location> LocENode(&LocE);
  Expected.Loc.push_back(LocENode);

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
  InfoNode<Reference> RCSharedENode(&RCSharedE);
  Expected.Children.Records.push_back(RCSharedENode);
  FunctionInfo FE1;
  FE1.Name = "OneFunction";
  FE1.USR = NonEmptySID;
  InfoNode<FunctionInfo> FE1Node(&FE1);
  Expected.Children.Functions.push_back(FE1Node);

  FunctionInfo FE2;
  FE2.Name = "TwoFunction";
  InfoNode<FunctionInfo> FE2Node(&FE2);
  Expected.Children.Functions.push_back(FE2Node);

  EnumInfo EE1;
  EE1.Name = "OneEnum";
  EE1.USR = NonEmptySID;
  InfoNode<EnumInfo> EE1Node(&EE1);
  Expected.Children.Enums.push_back(EE1Node);

  EnumInfo EE2;
  EE2.Name = "TwoEnum";
  InfoNode<EnumInfo> EE2Node(&EE2);
  Expected.Children.Enums.push_back(EE2Node);

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
  InfoNode<Location> Loc1Node(&Loc1);
  One.Loc.push_back(Loc1Node);

  One.IsMethod = true;
  One.Parent = Reference(EmptySID, "Parent", InfoType::IT_namespace);

  CommentInfo OneText[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, "This is a text comment.")};
  CommentInfo OnePara[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, OneText)};
  CommentInfo TopOne(CommentKind::CK_FullComment, OnePara);
  InfoNode<CommentInfo> TopOneNode(&TopOne);
  One.Description.push_back(TopOneNode);

  FunctionInfo Two;
  Two.Name = "f";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Location Loc2(12, 12, "test.cpp");
  InfoNode<Location> Loc2Node(&Loc2);
  Two.Loc.push_back(Loc2Node);

  Two.ReturnType = TypeInfo("void");
  FieldTypeInfo P2(TypeInfo("int"), "P");
  FieldTypeInfo Params2[] = {std::move(P2)};
  Two.Params = llvm::ArrayRef(Params2);

  CommentInfo TwoText[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, "This is a text comment.")};
  CommentInfo TwoPara[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, TwoText)};
  CommentInfo TopTwo(CommentKind::CK_FullComment, TwoPara);
  InfoNode<CommentInfo> TopTwoNode(&TopTwo);
  Two.Description.push_back(TopTwoNode);

  OwningPtrVec<Info> Infos;
  Infos.push_back(&One);
  Infos.push_back(&Two);

  FunctionInfo Expected;
  Expected.Name = "f";
  Reference NsE[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Expected.Namespace = llvm::ArrayRef(NsE);

  Expected.DefLoc = Location(10, 10, "test.cpp");
  Location LocE(12, 12, "test.cpp");
  InfoNode<Location> LocENode(&LocE);
  Expected.Loc.push_back(LocENode);

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
  InfoNode<CommentInfo> TopENode(&TopE);
  Expected.Description.push_back(TopENode);

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
  InfoNode<Location> Loc1Node(&Loc1);
  One.Loc.push_back(Loc1Node);

  One.Scoped = true;

  EnumInfo Two;
  Two.Name = "e";
  Reference Ns2[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  Two.Namespace = llvm::ArrayRef(Ns2);

  Location Loc2(20, 20, "test.cpp");
  InfoNode<Location> Loc2Node(&Loc2);
  Two.Loc.push_back(Loc2Node);

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
  InfoNode<Location> LocE1Node(&LocE1);
  Expected.Loc.push_back(LocE1Node);
  Location LocE2(20, 20, "test.cpp");
  InfoNode<Location> LocE2Node(&LocE2);
  Expected.Loc.push_back(LocE2Node);

  EnumValueInfo EV_E[] = {EnumValueInfo("X"), EnumValueInfo("Y")};
  Expected.Members = llvm::ArrayRef(EV_E);
  Expected.Scoped = true;

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckEnumInfo(InfoAsEnum(&Expected), InfoAsEnum(Actual.get()));
}

} // namespace doc
} // namespace clang
