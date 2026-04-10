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
#include "clang/Basic/DiagnosticOptions.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

ClangDocContextTest::ClangDocContextTest()
    : DiagID(new DiagnosticIDs()),
      Diags(DiagID, DiagOpts, new IgnoringDiagConsumer()) {}

ClangDocContextTest::~ClangDocContextTest() = default;

ClangDocContext ClangDocContextTest::getClangDocContext(
    std::vector<std::string> UserStylesheets, StringRef RepositoryUrl,
    StringRef RepositoryLinePrefix, StringRef Base) {
  return ClangDocContext(nullptr, "test-project", false, "", "", RepositoryUrl,
                         RepositoryLinePrefix, Base, UserStylesheets, Diags,
                         OutputFormatTy::html, false);
}

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

void CheckCommentInfo(ArrayRef<CommentInfo> Expected,
                      ArrayRef<CommentInfo> Actual);
void CheckCommentInfo(const OwningVec<CommentInfo> &Expected,
                      const OwningVec<CommentInfo> &Actual);

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

void CheckCommentInfo(ArrayRef<CommentInfo> Expected,
                      ArrayRef<CommentInfo> Actual) {
  auto ItE = Expected.begin();
  auto ItA = Actual.begin();
  while (ItE != Expected.end() && ItA != Actual.end()) {
    CheckCommentInfo(*ItE, *ItA);
    ++ItE;
    ++ItA;
  }
  EXPECT_TRUE(ItE == Expected.end() && ItA == Actual.end());
}

void CheckCommentInfo(const OwningVec<CommentInfo> &Expected,
                      const OwningVec<CommentInfo> &Actual) {
  auto ItE = Expected.begin();
  auto ItA = Actual.begin();
  while (ItE != Expected.end() && ItA != Actual.end()) {
    CheckCommentInfo(*ItE, *ItA);
    ++ItE;
    ++ItA;
  }
  EXPECT_TRUE(ItE == Expected.end() && ItA == Actual.end());
}

void CheckReference(const Reference &Expected, const Reference &Actual) {
  EXPECT_EQ(Expected.Name, Actual.Name);
  EXPECT_EQ(Expected.RefType, Actual.RefType);
  EXPECT_EQ(Expected.Path, Actual.Path);
}

void CheckTypeInfo(const TypeInfo *Expected, const TypeInfo *Actual) {
  CheckReference(Expected->Type, Actual->Type);
}

void CheckFieldTypeInfo(const FieldTypeInfo *Expected,
                        const FieldTypeInfo *Actual) {
  CheckTypeInfo(Expected, Actual);
  EXPECT_EQ(Expected->Name, Actual->Name);
}

void CheckMemberTypeInfo(const MemberTypeInfo *Expected,
                         const MemberTypeInfo *Actual) {
  CheckFieldTypeInfo(Expected, Actual);
  EXPECT_EQ(Expected->Access, Actual->Access);
  CheckCommentInfo(Expected->Description, Actual->Description);
}

void CheckBaseInfo(const Info *Expected, const Info *Actual) {
  EXPECT_EQ(size_t(20), Actual->USR.size());
  EXPECT_EQ(Expected->Name, Actual->Name);
  EXPECT_EQ(Expected->Path, Actual->Path);
  auto ItN_E = Expected->Namespace.begin();
  auto ItN_A = Actual->Namespace.begin();
  while (ItN_E != Expected->Namespace.end() &&
         ItN_A != Actual->Namespace.end()) {
    CheckReference(*ItN_E, *ItN_A);
    ++ItN_E;
    ++ItN_A;
  }
  EXPECT_TRUE(ItN_E == Expected->Namespace.end() &&
              ItN_A == Actual->Namespace.end());
  CheckCommentInfo(Expected->Description, Actual->Description);
}

void CheckSymbolInfo(const SymbolInfo *Expected, const SymbolInfo *Actual) {
  CheckBaseInfo(Expected, Actual);
  EXPECT_EQ(Expected->DefLoc.has_value(), Actual->DefLoc.has_value());
  if (Expected->DefLoc && Actual->DefLoc.has_value()) {
    EXPECT_EQ(Expected->DefLoc->StartLineNumber,
              Actual->DefLoc->StartLineNumber);
    EXPECT_EQ(Expected->DefLoc->EndLineNumber, Actual->DefLoc->EndLineNumber);
    EXPECT_EQ(Expected->DefLoc->Filename, Actual->DefLoc->Filename);
  }
  auto ItE = Expected->Loc.begin();
  auto ItA = Actual->Loc.begin();
  while (ItE != Expected->Loc.end() && ItA != Actual->Loc.end()) {
    EXPECT_EQ(*ItE, *ItA);
    ++ItE;
    ++ItA;
  }
  EXPECT_TRUE(ItE == Expected->Loc.end() && ItA == Actual->Loc.end());
}

void CheckFunctionInfo(const FunctionInfo *Expected,
                       const FunctionInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->IsMethod, Actual->IsMethod);
  CheckReference(Expected->Parent, Actual->Parent);
  CheckTypeInfo(&Expected->ReturnType, &Actual->ReturnType);

  for (size_t Idx = 0; Idx < Expected->Params.size(); ++Idx) {
    EXPECT_EQ(Expected->Params[Idx], Actual->Params[Idx]);
  }

  EXPECT_EQ(Expected->Access, Actual->Access);
}

void CheckEnumInfo(const EnumInfo *Expected, const EnumInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->Scoped, Actual->Scoped);
  auto ItM_E = Expected->Members.begin();
  auto ItM_A = Actual->Members.begin();
  while (ItM_E != Expected->Members.end() && ItM_A != Actual->Members.end()) {
    EXPECT_EQ(*ItM_E, *ItM_A);
    ++ItM_E;
    ++ItM_A;
  }
  EXPECT_TRUE(ItM_E == Expected->Members.end() &&
              ItM_A == Actual->Members.end());
}

void CheckTypedefInfo(const TypedefInfo *Expected, const TypedefInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);
  EXPECT_EQ(Expected->IsUsing, Actual->IsUsing);
  CheckTypeInfo(&Expected->Underlying, &Actual->Underlying);
}

void CheckNamespaceInfo(const NamespaceInfo *Expected,
                        const NamespaceInfo *Actual) {
  CheckBaseInfo(Expected, Actual);

  ASSERT_EQ(Expected->Children.Namespaces.size(),
            Actual->Children.Namespaces.size());
  auto ItN_E = Expected->Children.Namespaces.begin();
  auto ItN_A = Actual->Children.Namespaces.begin();
  while (ItN_E != Expected->Children.Namespaces.end() &&
         ItN_A != Actual->Children.Namespaces.end()) {
    CheckReference(*ItN_E, *ItN_A);
    ++ItN_E;
    ++ItN_A;
  }
  EXPECT_TRUE(ItN_E == Expected->Children.Namespaces.end() &&
              ItN_A == Actual->Children.Namespaces.end());

  auto ItR_E = Expected->Children.Records.begin();
  auto ItR_A = Actual->Children.Records.begin();
  while (ItR_E != Expected->Children.Records.end() &&
         ItR_A != Actual->Children.Records.end()) {
    CheckReference(*ItR_E, *ItR_A);
    ++ItR_E;
    ++ItR_A;
  }
  EXPECT_TRUE(ItR_E == Expected->Children.Records.end() &&
              ItR_A == Actual->Children.Records.end());

  auto ItF_E = Expected->Children.Functions.begin();
  auto ItF_A = Actual->Children.Functions.begin();
  while (ItF_E != Expected->Children.Functions.end() &&
         ItF_A != Actual->Children.Functions.end()) {
    CheckFunctionInfo(&(*ItF_E), &(*ItF_A));
    ++ItF_E;
    ++ItF_A;
  }
  EXPECT_TRUE(ItF_E == Expected->Children.Functions.end() &&
              ItF_A == Actual->Children.Functions.end());

  auto ItEnum_E = Expected->Children.Enums.begin();
  auto ItEnum_A = Actual->Children.Enums.begin();
  while (ItEnum_E != Expected->Children.Enums.end() &&
         ItEnum_A != Actual->Children.Enums.end()) {
    CheckEnumInfo(&(*ItEnum_E), &(*ItEnum_A));
    ++ItEnum_E;
    ++ItEnum_A;
  }
  EXPECT_TRUE(ItEnum_E == Expected->Children.Enums.end() &&
              ItEnum_A == Actual->Children.Enums.end());
}

void CheckRecordInfo(const RecordInfo *Expected, const RecordInfo *Actual) {
  CheckSymbolInfo(Expected, Actual);

  EXPECT_EQ(Expected->TagType, Actual->TagType);

  EXPECT_EQ(Expected->IsTypeDef, Actual->IsTypeDef);

  auto ItM_E = Expected->Members.begin();
  auto ItM_A = Actual->Members.begin();
  while (ItM_E != Expected->Members.end() && ItM_A != Actual->Members.end()) {
    EXPECT_EQ(*ItM_E, *ItM_A);
    ++ItM_E;
    ++ItM_A;
  }
  EXPECT_TRUE(ItM_E == Expected->Members.end() &&
              ItM_A == Actual->Members.end());

  auto ItP_E = Expected->Parents.begin();
  auto ItP_A = Actual->Parents.begin();
  while (ItP_E != Expected->Parents.end() && ItP_A != Actual->Parents.end()) {
    CheckReference(*ItP_E, *ItP_A);
    ++ItP_E;
    ++ItP_A;
  }
  EXPECT_TRUE(ItP_E == Expected->Parents.end() &&
              ItP_A == Actual->Parents.end());

  auto ItVP_E = Expected->VirtualParents.begin();
  auto ItVP_A = Actual->VirtualParents.begin();
  while (ItVP_E != Expected->VirtualParents.end() &&
         ItVP_A != Actual->VirtualParents.end()) {
    CheckReference(*ItVP_E, *ItVP_A);
    ++ItVP_E;
    ++ItVP_A;
  }
  EXPECT_TRUE(ItVP_E == Expected->VirtualParents.end() &&
              ItVP_A == Actual->VirtualParents.end());

  auto ItB_E = Expected->Bases.begin();
  auto ItB_A = Actual->Bases.begin();
  while (ItB_E != Expected->Bases.end() && ItB_A != Actual->Bases.end()) {
    CheckBaseRecordInfo(&(*ItB_E), &(*ItB_A));
    ++ItB_E;
    ++ItB_A;
  }
  EXPECT_TRUE(ItB_E == Expected->Bases.end() && ItB_A == Actual->Bases.end());

  auto ItR_E = Expected->Children.Records.begin();
  auto ItR_A = Actual->Children.Records.begin();
  while (ItR_E != Expected->Children.Records.end() &&
         ItR_A != Actual->Children.Records.end()) {
    CheckReference(*ItR_E, *ItR_A);
    ++ItR_E;
    ++ItR_A;
  }
  EXPECT_TRUE(ItR_E == Expected->Children.Records.end() &&
              ItR_A == Actual->Children.Records.end());

  auto ItF_E = Expected->Children.Functions.begin();
  auto ItF_A = Actual->Children.Functions.begin();
  while (ItF_E != Expected->Children.Functions.end() &&
         ItF_A != Actual->Children.Functions.end()) {
    CheckFunctionInfo(&(*ItF_E), &(*ItF_A));
    ++ItF_E;
    ++ItF_A;
  }
  EXPECT_TRUE(ItF_E == Expected->Children.Functions.end() &&
              ItF_A == Actual->Children.Functions.end());

  auto ItEnum_E = Expected->Children.Enums.begin();
  auto ItEnum_A = Actual->Children.Enums.begin();
  while (ItEnum_E != Expected->Children.Enums.end() &&
         ItEnum_A != Actual->Children.Enums.end()) {
    CheckEnumInfo(&(*ItEnum_E), &(*ItEnum_A));
    ++ItEnum_E;
    ++ItEnum_A;
  }
  EXPECT_TRUE(ItEnum_E == Expected->Children.Enums.end() &&
              ItEnum_A == Actual->Children.Enums.end());
}

void CheckBaseRecordInfo(const BaseRecordInfo *Expected,
                         const BaseRecordInfo *Actual) {
  CheckRecordInfo(Expected, Actual);

  EXPECT_EQ(Expected->IsVirtual, Actual->IsVirtual);
  EXPECT_EQ(Expected->Access, Actual->Access);
  EXPECT_EQ(Expected->IsParent, Actual->IsParent);
}

void CheckIndex(const Index &Expected, const Index &Actual) {
  CheckReference(Expected, Actual);
  ASSERT_EQ(Expected.Children.size(), Actual.Children.size());
  for (auto &[_, C] : Expected.Children)
    CheckIndex(C, Actual.Children.find(llvm::toStringRef(C.USR))->second);
}

} // namespace doc
} // namespace clang
