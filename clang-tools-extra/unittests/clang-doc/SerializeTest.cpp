//===-- clang-doc/SerializeTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Serialize.h"
#include "ClangDocTest.h"
#include "Representation.h"
#include "clang/AST/Comment.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

class ClangDocSerializeTestVisitor
    : public RecursiveASTVisitor<ClangDocSerializeTestVisitor> {

  EmittedInfoList &EmittedInfos;
  bool Public;

  comments::FullComment *getComment(const NamedDecl *D) const {
    if (RawComment *Comment =
            D->getASTContext().getRawCommentForDeclNoCache(D)) {
      Comment->setAttached();
      return Comment->parse(D->getASTContext(), nullptr, D);
    }
    return nullptr;
  }

public:
  ClangDocSerializeTestVisitor(EmittedInfoList &EmittedInfos, bool Public)
      : EmittedInfos(EmittedInfos), Public(Public) {}

  template <typename T> bool mapDecl(const T *D) {
    auto I = serialize::emitInfo(D, getComment(D), /*Line=*/0,
                                 /*File=*/"test.cpp", Public);
    if (I.first)
      EmittedInfos.emplace_back(std::move(I.first));
    if (I.second)
      EmittedInfos.emplace_back(std::move(I.second));
    return true;
  }

  bool VisitNamespaceDecl(const NamespaceDecl *D) { return mapDecl(D); }

  bool VisitFunctionDecl(const FunctionDecl *D) {
    // Don't visit CXXMethodDecls twice
    if (dyn_cast<CXXMethodDecl>(D))
      return true;
    return mapDecl(D);
  }

  bool VisitCXXMethodDecl(const CXXMethodDecl *D) { return mapDecl(D); }

  bool VisitRecordDecl(const RecordDecl *D) { return mapDecl(D); }

  bool VisitEnumDecl(const EnumDecl *D) { return mapDecl(D); }
};

void ExtractInfosFromCode(StringRef Code, size_t NumExpectedInfos, bool Public,
                          EmittedInfoList &EmittedInfos) {
  auto ASTUnit = clang::tooling::buildASTFromCode(Code);
  auto TU = ASTUnit->getASTContext().getTranslationUnitDecl();
  ClangDocSerializeTestVisitor Visitor(EmittedInfos, Public);
  Visitor.TraverseTranslationUnitDecl(TU);
  ASSERT_EQ(NumExpectedInfos, EmittedInfos.size());
}

void ExtractInfosFromCodeWithArgs(StringRef Code, size_t NumExpectedInfos,
                                  bool Public, EmittedInfoList &EmittedInfos,
                                  std::vector<std::string> &Args) {
  auto ASTUnit = clang::tooling::buildASTFromCodeWithArgs(Code, Args);
  auto TU = ASTUnit->getASTContext().getTranslationUnitDecl();
  ClangDocSerializeTestVisitor Visitor(EmittedInfos, Public);
  Visitor.TraverseTranslationUnitDecl(TU);
  ASSERT_EQ(NumExpectedInfos, EmittedInfos.size());
}

// Test serialization of namespace declarations.
TEST(SerializeTest, emitNamespaceInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("namespace A { namespace B { void f() {} } }", 5,
                       /*Public=*/false, Infos);

  NamespaceInfo *A = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedA(EmptySID, "A");
  CheckNamespaceInfo(&ExpectedA, A);

  NamespaceInfo *B = InfoAsNamespace(Infos[2].get());
  NamespaceInfo ExpectedB(EmptySID, "B");
  ExpectedB.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);
  CheckNamespaceInfo(&ExpectedB, B);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[4].get());
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "f";
  F.ReturnType = TypeInfo(EmptySID, "void", InfoType::IT_default);
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  F.Namespace.emplace_back(EmptySID, "B", InfoType::IT_namespace);
  F.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);
  ExpectedBWithFunction.ChildFunctions.emplace_back(std::move(F));
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST(SerializeTest, emitAnonymousNamespaceInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("namespace { }", 2, /*Public=*/false, Infos);

  NamespaceInfo *A = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedA(EmptySID);
  ExpectedA.Name = "@nonymous_namespace";
  CheckNamespaceInfo(&ExpectedA, A);
}

// Test serialization of record declarations.
TEST(SerializeTest, emitRecordInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode(R"raw(class E {
public:
  E() {}
protected:
  void ProtectedMethod();
};
template <typename T>
struct F {
  void TemplateMethod();
};
template <>
void F<int>::TemplateMethod();
typedef struct {} G;)raw",
                       10, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, "E");
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *RecordWithEConstructor = InfoAsRecord(Infos[2].get());
  RecordInfo ExpectedRecordWithEConstructor(EmptySID);
  FunctionInfo EConstructor;
  EConstructor.Name = "E";
  EConstructor.Parent = Reference(EmptySID, "E", InfoType::IT_record);
  EConstructor.ReturnType = TypeInfo(EmptySID, "void", InfoType::IT_default);
  EConstructor.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  EConstructor.Namespace.emplace_back(EmptySID, "E", InfoType::IT_record);
  EConstructor.Access = AccessSpecifier::AS_public;
  EConstructor.IsMethod = true;
  ExpectedRecordWithEConstructor.ChildFunctions.emplace_back(
      std::move(EConstructor));
  CheckRecordInfo(&ExpectedRecordWithEConstructor, RecordWithEConstructor);

  RecordInfo *RecordWithMethod = InfoAsRecord(Infos[3].get());
  RecordInfo ExpectedRecordWithMethod(EmptySID);
  FunctionInfo Method;
  Method.Name = "ProtectedMethod";
  Method.Parent = Reference(EmptySID, "E", InfoType::IT_record);
  Method.ReturnType = TypeInfo(EmptySID, "void", InfoType::IT_default);
  Method.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  Method.Namespace.emplace_back(EmptySID, "E", InfoType::IT_record);
  Method.Access = AccessSpecifier::AS_protected;
  Method.IsMethod = true;
  ExpectedRecordWithMethod.ChildFunctions.emplace_back(std::move(Method));
  CheckRecordInfo(&ExpectedRecordWithMethod, RecordWithMethod);

  RecordInfo *F = InfoAsRecord(Infos[4].get());
  RecordInfo ExpectedF(EmptySID, "F");
  ExpectedF.TagType = TagTypeKind::TTK_Struct;
  ExpectedF.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedF, F);

  RecordInfo *RecordWithTemplateMethod = InfoAsRecord(Infos[6].get());
  RecordInfo ExpectedRecordWithTemplateMethod(EmptySID);
  FunctionInfo TemplateMethod;
  TemplateMethod.Name = "TemplateMethod";
  TemplateMethod.Parent = Reference(EmptySID, "F", InfoType::IT_record);
  TemplateMethod.ReturnType = TypeInfo(EmptySID, "void", InfoType::IT_default);
  TemplateMethod.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  TemplateMethod.Namespace.emplace_back(EmptySID, "F", InfoType::IT_record);
  TemplateMethod.Access = AccessSpecifier::AS_public;
  TemplateMethod.IsMethod = true;
  ExpectedRecordWithTemplateMethod.ChildFunctions.emplace_back(
      std::move(TemplateMethod));
  CheckRecordInfo(&ExpectedRecordWithTemplateMethod, RecordWithTemplateMethod);

  RecordInfo *TemplatedRecord = InfoAsRecord(Infos[7].get());
  RecordInfo ExpectedTemplatedRecord(EmptySID);
  FunctionInfo SpecializedTemplateMethod;
  SpecializedTemplateMethod.Name = "TemplateMethod";
  SpecializedTemplateMethod.Parent =
      Reference(EmptySID, "F", InfoType::IT_record);
  SpecializedTemplateMethod.ReturnType =
      TypeInfo(EmptySID, "void", InfoType::IT_default);
  SpecializedTemplateMethod.Loc.emplace_back(0,
                                             llvm::SmallString<16>{"test.cpp"});
  SpecializedTemplateMethod.Namespace.emplace_back(EmptySID, "F",
                                                   InfoType::IT_record);
  SpecializedTemplateMethod.Access = AccessSpecifier::AS_public;
  SpecializedTemplateMethod.IsMethod = true;
  ExpectedTemplatedRecord.ChildFunctions.emplace_back(
      std::move(SpecializedTemplateMethod));
  CheckRecordInfo(&ExpectedTemplatedRecord, TemplatedRecord);

  RecordInfo *G = InfoAsRecord(Infos[8].get());
  RecordInfo ExpectedG(EmptySID, "G");
  ExpectedG.TagType = TagTypeKind::TTK_Struct;
  ExpectedG.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedG.IsTypeDef = true;
  CheckRecordInfo(&ExpectedG, G);
}

// Test serialization of enum declarations.
TEST(SerializeTest, emitEnumInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("enum E { X, Y }; enum class G { A, B };", 2,
                       /*Public=*/false, Infos);

  NamespaceInfo *NamespaceWithEnum = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedNamespaceWithEnum(EmptySID);
  EnumInfo E;
  E.Name = "E";
  E.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  E.Members.emplace_back("X");
  E.Members.emplace_back("Y");
  ExpectedNamespaceWithEnum.ChildEnums.emplace_back(std::move(E));
  CheckNamespaceInfo(&ExpectedNamespaceWithEnum, NamespaceWithEnum);

  NamespaceInfo *NamespaceWithScopedEnum = InfoAsNamespace(Infos[1].get());
  NamespaceInfo ExpectedNamespaceWithScopedEnum(EmptySID);
  EnumInfo G;
  G.Name = "G";
  G.Scoped = true;
  G.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  G.Members.emplace_back("A");
  G.Members.emplace_back("B");
  ExpectedNamespaceWithScopedEnum.ChildEnums.emplace_back(std::move(G));
  CheckNamespaceInfo(&ExpectedNamespaceWithScopedEnum, NamespaceWithScopedEnum);
}

TEST(SerializeTest, emitUndefinedRecordInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("class E;", 2, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, "E");
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  ExpectedE.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedE, E);
}

TEST(SerializeTest, emitRecordMemberInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("struct E { int I; };", 2, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, "E");
  ExpectedE.TagType = TagTypeKind::TTK_Struct;
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.Members.emplace_back("int", "I", AccessSpecifier::AS_public);
  CheckRecordInfo(&ExpectedE, E);
}

TEST(SerializeTest, emitInternalRecordInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("class E { class G {}; };", 4, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, "E");
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *G = InfoAsRecord(Infos[2].get());
  RecordInfo ExpectedG(EmptySID, "G");
  ExpectedG.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedG.TagType = TagTypeKind::TTK_Class;
  ExpectedG.Namespace.emplace_back(EmptySID, "E", InfoType::IT_record);
  CheckRecordInfo(&ExpectedG, G);
}

TEST(SerializeTest, emitPublicAnonymousNamespaceInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("namespace { class A; }", 0, /*Public=*/true, Infos);
}

TEST(SerializeTest, emitPublicFunctionInternalInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("int F() { class G {}; return 0; };", 1, /*Public=*/true,
                       Infos);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "F";
  F.ReturnType = TypeInfo(EmptySID, "int", InfoType::IT_default);
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedBWithFunction.ChildFunctions.emplace_back(std::move(F));
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST(SerializeTest, emitInlinedFunctionInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("inline void F(int I) { };", 1, /*Public=*/true, Infos);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "F";
  F.ReturnType = TypeInfo(EmptySID, "void", InfoType::IT_default);
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  F.Params.emplace_back("int", "I");
  ExpectedBWithFunction.ChildFunctions.emplace_back(std::move(F));
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST(SerializeTest, ) {
  EmittedInfoList Infos;
  ExtractInfosFromCode(R"raw(class F {};
class G {} ;
class E : public F, virtual private G {};
template <typename T>
class H {} ;
class I : public H<int> {} ;)raw",
                       10, /*Public=*/false, Infos);

  RecordInfo *F = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedF(EmptySID, "F");
  ExpectedF.TagType = TagTypeKind::TTK_Class;
  ExpectedF.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedF, F);

  RecordInfo *G = InfoAsRecord(Infos[2].get());
  RecordInfo ExpectedG(EmptySID, "G");
  ExpectedG.TagType = TagTypeKind::TTK_Class;
  ExpectedG.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedG, G);

  RecordInfo *E = InfoAsRecord(Infos[4].get());
  RecordInfo ExpectedE(EmptySID, "E");
  ExpectedE.Parents.emplace_back(EmptySID, "F", InfoType::IT_record);
  ExpectedE.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *H = InfoAsRecord(Infos[6].get());
  RecordInfo ExpectedH(EmptySID, "H");
  ExpectedH.TagType = TagTypeKind::TTK_Class;
  ExpectedH.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedH, H);

  RecordInfo *I = InfoAsRecord(Infos[8].get());
  RecordInfo ExpectedI(EmptySID, "I");
  ExpectedI.Parents.emplace_back(EmptySID, "H<int>", InfoType::IT_record);
  ExpectedI.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedI.TagType = TagTypeKind::TTK_Class;
  CheckRecordInfo(&ExpectedI, I);
}

TEST(SerializeTest, emitModulePublicLFunctions) {
  EmittedInfoList Infos;
  std::vector<std::string> Args;
  Args.push_back("-fmodules-ts");
  ExtractInfosFromCodeWithArgs(R"raw(export module M;
int moduleFunction(int x);
static int staticModuleFunction(int x);
export double exportedModuleFunction(double y);)raw",
                               2, /*Public=*/true, Infos, Args);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "moduleFunction";
  F.ReturnType = TypeInfo(EmptySID, "int", InfoType::IT_default);
  F.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  F.Params.emplace_back("int", "x");
  ExpectedBWithFunction.ChildFunctions.emplace_back(std::move(F));
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);

  NamespaceInfo *BWithExportedFunction = InfoAsNamespace(Infos[1].get());
  NamespaceInfo ExpectedBWithExportedFunction(EmptySID);
  FunctionInfo ExportedF;
  ExportedF.Name = "exportedModuleFunction";
  ExportedF.ReturnType = TypeInfo(EmptySID, "double", InfoType::IT_default);
  ExportedF.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  ExportedF.Params.emplace_back("double", "y");
  ExpectedBWithExportedFunction.ChildFunctions.emplace_back(
      std::move(ExportedF));
  CheckNamespaceInfo(&ExpectedBWithExportedFunction, BWithExportedFunction);
}

// Test serialization of child records in namespaces and other records
TEST(SerializeTest, emitChildRecords) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("class A { class B {}; }; namespace { class C {}; } ", 8,
                       /*Public=*/false, Infos);

  NamespaceInfo *ParentA = InfoAsNamespace(Infos[1].get());
  NamespaceInfo ExpectedParentA(EmptySID);
  ExpectedParentA.ChildRecords.emplace_back(EmptySID, "A", InfoType::IT_record);
  CheckNamespaceInfo(&ExpectedParentA, ParentA);

  RecordInfo *ParentB = InfoAsRecord(Infos[3].get());
  RecordInfo ExpectedParentB(EmptySID);
  ExpectedParentB.ChildRecords.emplace_back(EmptySID, "B", InfoType::IT_record);
  CheckRecordInfo(&ExpectedParentB, ParentB);

  NamespaceInfo *ParentC = InfoAsNamespace(Infos[7].get());
  NamespaceInfo ExpectedParentC(EmptySID);
  ExpectedParentC.ChildRecords.emplace_back(EmptySID, "C", InfoType::IT_record);
  CheckNamespaceInfo(&ExpectedParentC, ParentC);
}

// Test serialization of child namespaces
TEST(SerializeTest, emitChildNamespaces) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("namespace A { namespace B { } }", 4, /*Public=*/false,
                       Infos);

  NamespaceInfo *ParentA = InfoAsNamespace(Infos[1].get());
  NamespaceInfo ExpectedParentA(EmptySID);
  ExpectedParentA.ChildNamespaces.emplace_back(EmptySID, "A",
                                               InfoType::IT_namespace);
  CheckNamespaceInfo(&ExpectedParentA, ParentA);

  NamespaceInfo *ParentB = InfoAsNamespace(Infos[3].get());
  NamespaceInfo ExpectedParentB(EmptySID);
  ExpectedParentB.ChildNamespaces.emplace_back(EmptySID, "B",
                                               InfoType::IT_namespace);
  CheckNamespaceInfo(&ExpectedParentB, ParentB);
}

} // namespace doc
} // end namespace clang
