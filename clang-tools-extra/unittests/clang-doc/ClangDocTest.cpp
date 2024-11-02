//===-- clang-doc/ClangDocTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocTest.h"
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

TypedefInfo *InfoAsTypedef(Info *I) {
  assert(I->IT == InfoType::IT_typedef);
  return static_cast<TypedefInfo *>(I);
}

void CheckCommentInfo(const std::vector<CommentInfo> &Expected,
                      const std::vector<CommentInfo> &Actual);
void CheckCommentInfo(const std::vector<std::unique_ptr<CommentInfo>> &Expected,
                      const std::vector<std::unique_ptr<CommentInfo>> &Actual);

void CheckCommentInfo(const CommentInfo &Expected, const CommentInfo &Actual) {
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

  CheckCommentInfo(Expected.Children, Actual.Children);
}

void CheckCommentInfo(const std::vector<CommentInfo> &Expected,
                      const std::vector<CommentInfo> &Actual) {
  ASSERT_EQ(Expected.size(), Actual.size());
  for (size_t Idx = 0; Idx < Actual.size(); ++Idx)
    CheckCommentInfo(Expected[Idx], Actual[Idx]);
}

void CheckCommentInfo(const std::vector<std::unique_ptr<CommentInfo>> &Expected,
                      const std::vector<std::unique_ptr<CommentInfo>> &Actual) {
  ASSERT_EQ(Expected.size(), Actual.size());
  for (size_t Idx = 0; Idx < Actual.size(); ++Idx)
    CheckCommentInfo(*Expected[Idx], *Actual[Idx]);
}

void CheckReference(Reference &Expected, Reference &Actual) {
  EXPECT_EQ(Expected.Name, Actual.Name);
  EXPECT_EQ(Expected.RefType, Actual.RefType);
  EXPECT_EQ(Expected.Path, Actual.Path);
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
  CheckCommentInfo(Expected->Description, Actual->Description);
}

void CheckBaseInfo(Info *Expected, Info *Actual) {
  EXPECT_EQ(size_t(20), Actual->USR.size());
  EXPECT_EQ(Expected->Name, Actual->Name);
  EXPECT_EQ(Expected->Path, Actual->Path);
  ASSERT_EQ(Expected->Namespace.size(), Actual->Namespace.size());
  for (size_t Idx = 0; Idx < Actual->Namespace.size(); ++Idx)
    CheckReference(Expected->Namespace[Idx], Actual->Namespace[Idx]);
  CheckCommentInfo(Expected->Description, Actual->Description);
}

void CheckSymbolInfo(SymbolInfo *Expected, SymbolInfo *Actual) {
  CheckBaseInfo(Expected, Actual);
  EXPECT_EQ(Expected->DefLoc.has_value(), Actual->DefLoc.has_value());
  if (Expected->DefLoc && Actual->DefLoc.has_value()) {
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

void CheckTypedefInfo(TypedefInfo *Expected, TypedefInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);
  EXPECT_EQ(Expected->IsUsing, Actual->IsUsing);
  CheckTypeInfo(&Expected->Underlying, &Actual->Underlying);
}

void CheckNamespaceInfo(NamespaceInfo *Expected, NamespaceInfo *Actual) {
  CheckBaseInfo(Expected, Actual);

  ASSERT_EQ(Expected->Children.Namespaces.size(),
            Actual->Children.Namespaces.size());
  for (size_t Idx = 0; Idx < Actual->Children.Namespaces.size(); ++Idx)
    CheckReference(Expected->Children.Namespaces[Idx],
                   Actual->Children.Namespaces[Idx]);

  ASSERT_EQ(Expected->Children.Records.size(), Actual->Children.Records.size());
  for (size_t Idx = 0; Idx < Actual->Children.Records.size(); ++Idx)
    CheckReference(Expected->Children.Records[Idx],
                   Actual->Children.Records[Idx]);

  ASSERT_EQ(Expected->Children.Functions.size(),
            Actual->Children.Functions.size());
  for (size_t Idx = 0; Idx < Actual->Children.Functions.size(); ++Idx)
    CheckFunctionInfo(&Expected->Children.Functions[Idx],
                      &Actual->Children.Functions[Idx]);

  ASSERT_EQ(Expected->Children.Enums.size(), Actual->Children.Enums.size());
  for (size_t Idx = 0; Idx < Actual->Children.Enums.size(); ++Idx)
    CheckEnumInfo(&Expected->Children.Enums[Idx], &Actual->Children.Enums[Idx]);
}

void CheckRecordInfo(RecordInfo *Expected, RecordInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->TagType, Actual->TagType);

  EXPECT_EQ(Expected->IsTypeDef, Actual->IsTypeDef);

  ASSERT_EQ(Expected->Members.size(), Actual->Members.size());
  for (size_t Idx = 0; Idx < Actual->Members.size(); ++Idx)
    EXPECT_EQ(Expected->Members[Idx], Actual->Members[Idx]);

  ASSERT_EQ(Expected->Parents.size(), Actual->Parents.size());
  for (size_t Idx = 0; Idx < Actual->Parents.size(); ++Idx)
    CheckReference(Expected->Parents[Idx], Actual->Parents[Idx]);

  ASSERT_EQ(Expected->VirtualParents.size(), Actual->VirtualParents.size());
  for (size_t Idx = 0; Idx < Actual->VirtualParents.size(); ++Idx)
    CheckReference(Expected->VirtualParents[Idx], Actual->VirtualParents[Idx]);

  ASSERT_EQ(Expected->Bases.size(), Actual->Bases.size());
  for (size_t Idx = 0; Idx < Actual->Bases.size(); ++Idx)
    CheckBaseRecordInfo(&Expected->Bases[Idx], &Actual->Bases[Idx]);

  ASSERT_EQ(Expected->Children.Records.size(), Actual->Children.Records.size());
  for (size_t Idx = 0; Idx < Actual->Children.Records.size(); ++Idx)
    CheckReference(Expected->Children.Records[Idx],
                   Actual->Children.Records[Idx]);

  ASSERT_EQ(Expected->Children.Functions.size(),
            Actual->Children.Functions.size());
  for (size_t Idx = 0; Idx < Actual->Children.Functions.size(); ++Idx)
    CheckFunctionInfo(&Expected->Children.Functions[Idx],
                      &Actual->Children.Functions[Idx]);

  ASSERT_EQ(Expected->Children.Enums.size(), Actual->Children.Enums.size());
  for (size_t Idx = 0; Idx < Actual->Children.Enums.size(); ++Idx)
    CheckEnumInfo(&Expected->Children.Enums[Idx], &Actual->Children.Enums[Idx]);
}

void CheckBaseRecordInfo(BaseRecordInfo *Expected, BaseRecordInfo *Actual) {
  CheckRecordInfo(Expected, Actual);

  EXPECT_EQ(Expected->IsVirtual, Actual->IsVirtual);
  EXPECT_EQ(Expected->Access, Actual->Access);
  EXPECT_EQ(Expected->IsParent, Actual->IsParent);
}

void CheckIndex(Index &Expected, Index &Actual) {
  CheckReference(Expected, Actual);
  ASSERT_EQ(Expected.Children.size(), Actual.Children.size());
  for (size_t Idx = 0; Idx < Actual.Children.size(); ++Idx)
    CheckIndex(Expected.Children[Idx], Actual.Children[Idx]);
}

} // namespace doc
} // namespace clang
