//===-- clang-doc/ClangDocTest.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Representation.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

NamespaceInfo *InfoAsNamespace(Info *I) {
  assert(I->IT == InfoType::IT_namespace);
  return static_cast<NamespaceInfo *>(I);
}

RecordInfo *InfoAsRecord(Info *I) {
  assert(I->IT == InfoType::IT_record);
  return static_cast<RecordInfo *>(I);
}

FunctionInfo *InfoAsFunction(Info *I) {
  assert(I->IT == InfoType::IT_function);
  return static_cast<FunctionInfo *>(I);
}

EnumInfo *InfoAsEnum(Info *I) {
  assert(I->IT == InfoType::IT_enum);
  return static_cast<EnumInfo *>(I);
}

void CheckCommentInfo(CommentInfo &Expected, CommentInfo &Actual) {
  EXPECT_EQ(Expected.Kind, Actual.Kind);
  EXPECT_EQ(Expected.Text, Actual.Text);
  EXPECT_EQ(Expected.Name, Actual.Name);
  EXPECT_EQ(Expected.Direction, Actual.Direction);
  EXPECT_EQ(Expected.ParamName, Actual.ParamName);
  EXPECT_EQ(Expected.CloseName, Actual.CloseName);
  EXPECT_EQ(Expected.SelfClosing, Actual.SelfClosing);
  EXPECT_EQ(Expected.Explicit, Actual.Explicit);

  ASSERT_EQ(Expected.AttrKeys.size(), Actual.AttrKeys.size());
  for (size_t Idx = 0; Idx < Actual.AttrKeys.size(); ++Idx)
    EXPECT_EQ(Expected.AttrKeys[Idx], Actual.AttrKeys[Idx]);

  ASSERT_EQ(Expected.AttrValues.size(), Actual.AttrValues.size());
  for (size_t Idx = 0; Idx < Actual.AttrValues.size(); ++Idx)
    EXPECT_EQ(Expected.AttrValues[Idx], Actual.AttrValues[Idx]);

  ASSERT_EQ(Expected.Args.size(), Actual.Args.size());
  for (size_t Idx = 0; Idx < Actual.Args.size(); ++Idx)
    EXPECT_EQ(Expected.Args[Idx], Actual.Args[Idx]);

  ASSERT_EQ(Expected.Children.size(), Actual.Children.size());
  for (size_t Idx = 0; Idx < Actual.Children.size(); ++Idx)
    CheckCommentInfo(*Expected.Children[Idx], *Actual.Children[Idx]);
}

void CheckReference(Reference &Expected, Reference &Actual) {
  EXPECT_EQ(Expected.Name, Actual.Name);
  EXPECT_EQ(Expected.RefType, Actual.RefType);
}

void CheckTypeInfo(TypeInfo *Expected, TypeInfo *Actual) {
  CheckReference(Expected->Type, Actual->Type);
}

void CheckFieldTypeInfo(FieldTypeInfo *Expected, FieldTypeInfo *Actual) {
  CheckTypeInfo(Expected, Actual);
  EXPECT_EQ(Expected->Name, Actual->Name);
}

void CheckMemberTypeInfo(MemberTypeInfo *Expected, MemberTypeInfo *Actual) {
  CheckFieldTypeInfo(Expected, Actual);
  EXPECT_EQ(Expected->Access, Actual->Access);
}

void CheckBaseInfo(Info *Expected, Info *Actual) {
  EXPECT_EQ(size_t(20), Actual->USR.size());
  EXPECT_EQ(Expected->Name, Actual->Name);
  ASSERT_EQ(Expected->Namespace.size(), Actual->Namespace.size());
  for (size_t Idx = 0; Idx < Actual->Namespace.size(); ++Idx)
    CheckReference(Expected->Namespace[Idx], Actual->Namespace[Idx]);
  ASSERT_EQ(Expected->Description.size(), Actual->Description.size());
  for (size_t Idx = 0; Idx < Actual->Description.size(); ++Idx)
    CheckCommentInfo(Expected->Description[Idx], Actual->Description[Idx]);
}

void CheckSymbolInfo(SymbolInfo *Expected, SymbolInfo *Actual) {
  CheckBaseInfo(Expected, Actual);
  EXPECT_EQ(Expected->DefLoc.hasValue(), Actual->DefLoc.hasValue());
  if (Expected->DefLoc.hasValue() && Actual->DefLoc.hasValue()) {
    EXPECT_EQ(Expected->DefLoc->LineNumber, Actual->DefLoc->LineNumber);
    EXPECT_EQ(Expected->DefLoc->Filename, Actual->DefLoc->Filename);
  }
  ASSERT_EQ(Expected->Loc.size(), Actual->Loc.size());
  for (size_t Idx = 0; Idx < Actual->Loc.size(); ++Idx)
    EXPECT_EQ(Expected->Loc[Idx], Actual->Loc[Idx]);
}

void CheckFunctionInfo(FunctionInfo *Expected, FunctionInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->IsMethod, Actual->IsMethod);
  CheckReference(Expected->Parent, Actual->Parent);
  CheckTypeInfo(&Expected->ReturnType, &Actual->ReturnType);

  ASSERT_EQ(Expected->Params.size(), Actual->Params.size());
  for (size_t Idx = 0; Idx < Actual->Params.size(); ++Idx)
    EXPECT_EQ(Expected->Params[Idx], Actual->Params[Idx]);

  EXPECT_EQ(Expected->Access, Actual->Access);
}

void CheckEnumInfo(EnumInfo *Expected, EnumInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->Scoped, Actual->Scoped);
  ASSERT_EQ(Expected->Members.size(), Actual->Members.size());
  for (size_t Idx = 0; Idx < Actual->Members.size(); ++Idx)
    EXPECT_EQ(Expected->Members[Idx], Actual->Members[Idx]);
}

void CheckNamespaceInfo(NamespaceInfo *Expected, NamespaceInfo *Actual) {
  CheckBaseInfo(Expected, Actual);

  ASSERT_EQ(Expected->ChildNamespaces.size(), Actual->ChildNamespaces.size());
  for (size_t Idx = 0; Idx < Actual->ChildNamespaces.size(); ++Idx)
    EXPECT_EQ(Expected->ChildNamespaces[Idx], Actual->ChildNamespaces[Idx]);

  ASSERT_EQ(Expected->ChildRecords.size(), Actual->ChildRecords.size());
  for (size_t Idx = 0; Idx < Actual->ChildRecords.size(); ++Idx)
    EXPECT_EQ(Expected->ChildRecords[Idx], Actual->ChildRecords[Idx]);

  ASSERT_EQ(Expected->ChildFunctions.size(), Actual->ChildFunctions.size());
  for (size_t Idx = 0; Idx < Actual->ChildFunctions.size(); ++Idx)
    CheckFunctionInfo(&Expected->ChildFunctions[Idx],
                      &Actual->ChildFunctions[Idx]);

  ASSERT_EQ(Expected->ChildEnums.size(), Actual->ChildEnums.size());
  for (size_t Idx = 0; Idx < Actual->ChildEnums.size(); ++Idx)
    CheckEnumInfo(&Expected->ChildEnums[Idx], &Actual->ChildEnums[Idx]);
}

void CheckRecordInfo(RecordInfo *Expected, RecordInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->TagType, Actual->TagType);

  ASSERT_EQ(Expected->Members.size(), Actual->Members.size());
  for (size_t Idx = 0; Idx < Actual->Members.size(); ++Idx)
    EXPECT_EQ(Expected->Members[Idx], Actual->Members[Idx]);

  ASSERT_EQ(Expected->Parents.size(), Actual->Parents.size());
  for (size_t Idx = 0; Idx < Actual->Parents.size(); ++Idx)
    CheckReference(Expected->Parents[Idx], Actual->Parents[Idx]);

  ASSERT_EQ(Expected->VirtualParents.size(), Actual->VirtualParents.size());
  for (size_t Idx = 0; Idx < Actual->VirtualParents.size(); ++Idx)
    CheckReference(Expected->VirtualParents[Idx], Actual->VirtualParents[Idx]);

  ASSERT_EQ(Expected->ChildRecords.size(), Actual->ChildRecords.size());
  for (size_t Idx = 0; Idx < Actual->ChildRecords.size(); ++Idx)
    EXPECT_EQ(Expected->ChildRecords[Idx], Actual->ChildRecords[Idx]);

  ASSERT_EQ(Expected->ChildFunctions.size(), Actual->ChildFunctions.size());
  for (size_t Idx = 0; Idx < Actual->ChildFunctions.size(); ++Idx)
    CheckFunctionInfo(&Expected->ChildFunctions[Idx],
                      &Actual->ChildFunctions[Idx]);

  ASSERT_EQ(Expected->ChildEnums.size(), Actual->ChildEnums.size());
  for (size_t Idx = 0; Idx < Actual->ChildEnums.size(); ++Idx)
    CheckEnumInfo(&Expected->ChildEnums[Idx], &Actual->ChildEnums[Idx]);
}

} // namespace doc
} // namespace clang
