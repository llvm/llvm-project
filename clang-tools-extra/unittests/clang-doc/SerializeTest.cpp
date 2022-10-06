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
                                 /*File=*/"test.cpp", true, Public);
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

// Constructs a comment definition as the parser would for one comment line.
/* TODO uncomment this when the missing comment is fixed in emitRecordInfo and
   the code that calls this is re-enabled.
CommentInfo MakeOneLineCommentInfo(const std::string &Text) {
  CommentInfo TopComment;
  TopComment.Kind = "FullComment";
  TopComment.Children.emplace_back(std::make_unique<CommentInfo>());

  CommentInfo *Brief = TopComment.Children.back().get();
  Brief->Kind = "ParagraphComment";

  Brief->Children.emplace_back(std::make_unique<CommentInfo>());
  Brief->Children.back()->Kind = "TextComment";
  Brief->Children.back()->Name = "ParagraphComment";
  Brief->Children.back()->Text = Text;

  return TopComment;
}
*/

// Test serialization of namespace declarations.
TEST(SerializeTest, emitNamespaceInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("namespace A { namespace B { void f() {} } }", 5,
                       /*Public=*/false, Infos);

  NamespaceInfo *A = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedA(EmptySID, "A");
  CheckNamespaceInfo(&ExpectedA, A);

  NamespaceInfo *B = InfoAsNamespace(Infos[2].get());
  NamespaceInfo ExpectedB(EmptySID, /*Name=*/"B", /*Path=*/"A");
  ExpectedB.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);
  CheckNamespaceInfo(&ExpectedB, B);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[4].get());
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "f";
  F.ReturnType = TypeInfo("void");
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  F.Namespace.emplace_back(EmptySID, "B", InfoType::IT_namespace);
  F.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);
  F.Access = AccessSpecifier::AS_none;
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

  // Some docs.
  int value;
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
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  ExpectedE.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.Members.emplace_back(TypeInfo("int"), "value",
                                 AccessSpecifier::AS_public);
  // TODO the data member should have the docstring on it:
  //ExpectedE.Members.back().Description.push_back(MakeOneLineCommentInfo(" Some docs"));
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *RecordWithEConstructor = InfoAsRecord(Infos[2].get());
  RecordInfo ExpectedRecordWithEConstructor(EmptySID);
  FunctionInfo EConstructor;
  EConstructor.Name = "E";
  EConstructor.Parent = Reference(EmptySID, "E", InfoType::IT_record);
  EConstructor.ReturnType = TypeInfo("void");
  EConstructor.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  EConstructor.Namespace.emplace_back(EmptySID, "E", InfoType::IT_record);
  EConstructor.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                      InfoType::IT_namespace);
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
  Method.ReturnType = TypeInfo("void");
  Method.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  Method.Namespace.emplace_back(EmptySID, "E", InfoType::IT_record);
  Method.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                InfoType::IT_namespace);
  Method.Access = AccessSpecifier::AS_protected;
  Method.IsMethod = true;
  ExpectedRecordWithMethod.ChildFunctions.emplace_back(std::move(Method));
  CheckRecordInfo(&ExpectedRecordWithMethod, RecordWithMethod);

  RecordInfo *F = InfoAsRecord(Infos[4].get());
  RecordInfo ExpectedF(EmptySID, /*Name=*/"F", /*Path=*/"GlobalNamespace");
  ExpectedF.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedF.TagType = TagTypeKind::TTK_Struct;
  ExpectedF.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedF, F);

  RecordInfo *RecordWithTemplateMethod = InfoAsRecord(Infos[6].get());
  RecordInfo ExpectedRecordWithTemplateMethod(EmptySID);
  FunctionInfo TemplateMethod;
  TemplateMethod.Name = "TemplateMethod";
  TemplateMethod.Parent = Reference(EmptySID, "F", InfoType::IT_record);
  TemplateMethod.ReturnType = TypeInfo("void");
  TemplateMethod.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  TemplateMethod.Namespace.emplace_back(EmptySID, "F", InfoType::IT_record);
  TemplateMethod.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                        InfoType::IT_namespace);
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
  SpecializedTemplateMethod.ReturnType = TypeInfo("void");
  SpecializedTemplateMethod.Loc.emplace_back(0,
                                             llvm::SmallString<16>{"test.cpp"});
  SpecializedTemplateMethod.Namespace.emplace_back(EmptySID, "F",
                                                   InfoType::IT_record);
  SpecializedTemplateMethod.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                                   InfoType::IT_namespace);
  SpecializedTemplateMethod.Access = AccessSpecifier::AS_public;
  SpecializedTemplateMethod.IsMethod = true;
  ExpectedTemplatedRecord.ChildFunctions.emplace_back(
      std::move(SpecializedTemplateMethod));
  CheckRecordInfo(&ExpectedTemplatedRecord, TemplatedRecord);

  RecordInfo *G = InfoAsRecord(Infos[8].get());
  RecordInfo ExpectedG(EmptySID, /*Name=*/"G", /*Path=*/"GlobalNamespace");
  ExpectedG.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
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
  E.Members.emplace_back("X", "0");
  E.Members.emplace_back("Y", "1");
  ExpectedNamespaceWithEnum.ChildEnums.emplace_back(std::move(E));
  CheckNamespaceInfo(&ExpectedNamespaceWithEnum, NamespaceWithEnum);

  NamespaceInfo *NamespaceWithScopedEnum = InfoAsNamespace(Infos[1].get());
  NamespaceInfo ExpectedNamespaceWithScopedEnum(EmptySID);
  EnumInfo G;
  G.Name = "G";
  G.Scoped = true;
  G.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  G.Members.emplace_back("A", "0");
  G.Members.emplace_back("B", "1");
  ExpectedNamespaceWithScopedEnum.ChildEnums.emplace_back(std::move(G));
  CheckNamespaceInfo(&ExpectedNamespaceWithScopedEnum, NamespaceWithScopedEnum);
}

TEST(SerializeTest, emitUndefinedRecordInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("class E;", 2, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  ExpectedE.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  ExpectedE.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedE, E);
}

TEST(SerializeTest, emitRecordMemberInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("struct E { int I; };", 2, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  ExpectedE.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedE.TagType = TagTypeKind::TTK_Struct;
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.Members.emplace_back(TypeInfo("int"), "I",
                                 AccessSpecifier::AS_public);
  CheckRecordInfo(&ExpectedE, E);
}

TEST(SerializeTest, emitInternalRecordInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode("class E { class G {}; };", 4, /*Public=*/false, Infos);

  RecordInfo *E = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  ExpectedE.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *G = InfoAsRecord(Infos[2].get());
  llvm::SmallString<128> ExpectedGPath("GlobalNamespace/E");
  llvm::sys::path::native(ExpectedGPath);
  RecordInfo ExpectedG(EmptySID, /*Name=*/"G", /*Path=*/ExpectedGPath);
  ExpectedG.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedG.TagType = TagTypeKind::TTK_Class;
  ExpectedG.Namespace.emplace_back(EmptySID, "E", InfoType::IT_record);
  ExpectedG.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
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
  F.ReturnType = TypeInfo("int");
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  F.Access = AccessSpecifier::AS_none;
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
  F.ReturnType = TypeInfo("void");
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  F.Params.emplace_back(TypeInfo("int"), "I");
  F.Access = AccessSpecifier::AS_none;
  ExpectedBWithFunction.ChildFunctions.emplace_back(std::move(F));
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST(SerializeTest, emitInheritedRecordInfo) {
  EmittedInfoList Infos;
  ExtractInfosFromCode(R"raw(class F { protected: void set(int N); };
class G { public: int get() { return 1; } protected: int I; };
class E : public F, virtual private G {};
class H : private E {};
template <typename T>
class I {} ;
class J : public I<int> {} ;)raw",
                       14, /*Public=*/false, Infos);

  RecordInfo *F = InfoAsRecord(Infos[0].get());
  RecordInfo ExpectedF(EmptySID, /*Name=*/"F", /*Path=*/"GlobalNamespace");
  ExpectedF.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedF.TagType = TagTypeKind::TTK_Class;
  ExpectedF.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedF, F);

  RecordInfo *G = InfoAsRecord(Infos[3].get());
  RecordInfo ExpectedG(EmptySID, /*Name=*/"G", /*Path=*/"GlobalNamespace");
  ExpectedG.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedG.TagType = TagTypeKind::TTK_Class;
  ExpectedG.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedG.Members.emplace_back(TypeInfo("int"), "I",
                                 AccessSpecifier::AS_protected);
  CheckRecordInfo(&ExpectedG, G);

  RecordInfo *E = InfoAsRecord(Infos[6].get());
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  ExpectedE.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedE.Parents.emplace_back(EmptySID, /*Name=*/"F", InfoType::IT_record,
                                 /*Path*=*/"GlobalNamespace");
  ExpectedE.VirtualParents.emplace_back(
      EmptySID, /*Name=*/"G", InfoType::IT_record, /*Path*=*/"GlobalNamespace");
  ExpectedE.Bases.emplace_back(EmptySID, /*Name=*/"F",
                               /*Path=*/"GlobalNamespace", false,
                               AccessSpecifier::AS_public, true);
  FunctionInfo FunctionSet;
  FunctionSet.Name = "set";
  FunctionSet.ReturnType = TypeInfo("void");
  FunctionSet.Loc.emplace_back();
  FunctionSet.Params.emplace_back(TypeInfo("int"), "N");
  FunctionSet.Namespace.emplace_back(EmptySID, "F", InfoType::IT_record);
  FunctionSet.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                     InfoType::IT_namespace);
  FunctionSet.Access = AccessSpecifier::AS_protected;
  FunctionSet.IsMethod = true;
  ExpectedE.Bases.back().ChildFunctions.emplace_back(std::move(FunctionSet));
  ExpectedE.Bases.emplace_back(EmptySID, /*Name=*/"G",
                               /*Path=*/"GlobalNamespace", true,
                               AccessSpecifier::AS_private, true);
  FunctionInfo FunctionGet;
  FunctionGet.Name = "get";
  FunctionGet.ReturnType = TypeInfo("int");
  FunctionGet.DefLoc = Location();
  FunctionGet.Namespace.emplace_back(EmptySID, "G", InfoType::IT_record);
  FunctionGet.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                     InfoType::IT_namespace);
  FunctionGet.Access = AccessSpecifier::AS_private;
  FunctionGet.IsMethod = true;
  ExpectedE.Bases.back().ChildFunctions.emplace_back(std::move(FunctionGet));
  ExpectedE.Bases.back().Members.emplace_back(TypeInfo("int"), "I",
                                              AccessSpecifier::AS_private);
  ExpectedE.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedE.TagType = TagTypeKind::TTK_Class;
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *H = InfoAsRecord(Infos[8].get());
  RecordInfo ExpectedH(EmptySID, /*Name=*/"H", /*Path=*/"GlobalNamespace");
  ExpectedH.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedH.TagType = TagTypeKind::TTK_Class;
  ExpectedH.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedH.Parents.emplace_back(EmptySID, /*Name=*/"E", InfoType::IT_record,
                                 /*Path=*/"GlobalNamespace");
  ExpectedH.VirtualParents.emplace_back(
      EmptySID, /*Name=*/"G", InfoType::IT_record, /*Path=*/"GlobalNamespace");
  ExpectedH.Bases.emplace_back(EmptySID, /*Name=*/"E",
                               /*Path=*/"GlobalNamespace", false,
                               AccessSpecifier::AS_private, true);
  ExpectedH.Bases.emplace_back(EmptySID, /*Name=*/"F",
                               /*Path=*/"GlobalNamespace", false,
                               AccessSpecifier::AS_private, false);
  FunctionInfo FunctionSetNew;
  FunctionSetNew.Name = "set";
  FunctionSetNew.ReturnType = TypeInfo("void");
  FunctionSetNew.Loc.emplace_back();
  FunctionSetNew.Params.emplace_back(TypeInfo("int"), "N");
  FunctionSetNew.Namespace.emplace_back(EmptySID, "F", InfoType::IT_record);
  FunctionSetNew.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                        InfoType::IT_namespace);
  FunctionSetNew.Access = AccessSpecifier::AS_private;
  FunctionSetNew.IsMethod = true;
  ExpectedH.Bases.back().ChildFunctions.emplace_back(std::move(FunctionSetNew));
  ExpectedH.Bases.emplace_back(EmptySID, /*Name=*/"G",
                               /*Path=*/"GlobalNamespace", true,
                               AccessSpecifier::AS_private, false);
  FunctionInfo FunctionGetNew;
  FunctionGetNew.Name = "get";
  FunctionGetNew.ReturnType = TypeInfo("int");
  FunctionGetNew.DefLoc = Location();
  FunctionGetNew.Namespace.emplace_back(EmptySID, "G", InfoType::IT_record);
  FunctionGetNew.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                        InfoType::IT_namespace);
  FunctionGetNew.Access = AccessSpecifier::AS_private;
  FunctionGetNew.IsMethod = true;
  ExpectedH.Bases.back().ChildFunctions.emplace_back(std::move(FunctionGetNew));
  ExpectedH.Bases.back().Members.emplace_back(TypeInfo("int"), "I",
                                              AccessSpecifier::AS_private);
  CheckRecordInfo(&ExpectedH, H);

  RecordInfo *I = InfoAsRecord(Infos[10].get());
  RecordInfo ExpectedI(EmptySID, /*Name=*/"I", /*Path=*/"GlobalNamespace");
  ExpectedI.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedI.TagType = TagTypeKind::TTK_Class;
  ExpectedI.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  CheckRecordInfo(&ExpectedI, I);

  RecordInfo *J = InfoAsRecord(Infos[12].get());
  RecordInfo ExpectedJ(EmptySID, /*Name=*/"J", /*Path=*/"GlobalNamespace");
  ExpectedJ.Namespace.emplace_back(EmptySID, "GlobalNamespace",
                                   InfoType::IT_namespace);
  ExpectedJ.Parents.emplace_back(EmptySID, /*Name=*/"I<int>",
                                 InfoType::IT_record);
  ExpectedJ.Bases.emplace_back(EmptySID, /*Name=*/"I<int>",
                               /*Path=*/"GlobalNamespace", false,
                               AccessSpecifier::AS_public, true);
  ExpectedJ.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  ExpectedJ.TagType = TagTypeKind::TTK_Class;
  CheckRecordInfo(&ExpectedJ, J);
}

TEST(SerializeTest, emitModulePublicLFunctions) {
  EmittedInfoList Infos;
  std::vector<std::string> Args;
  Args.push_back("-fmodules-ts");
  ExtractInfosFromCodeWithArgs(R"raw(export module M;
int moduleFunction(int x, double d = 3.2 - 1.0);
static int staticModuleFunction(int x);
export double exportedModuleFunction(double y);)raw",
                               2, /*Public=*/true, Infos, Args);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0].get());
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "moduleFunction";
  F.ReturnType = TypeInfo("int");
  F.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  F.Params.emplace_back(TypeInfo("int"), "x");
  F.Params.emplace_back(TypeInfo("double"), "d");
  F.Params.back().DefaultValue = "3.2 - 1.0";
  F.Access = AccessSpecifier::AS_none;
  ExpectedBWithFunction.ChildFunctions.emplace_back(std::move(F));
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);

  NamespaceInfo *BWithExportedFunction = InfoAsNamespace(Infos[1].get());
  NamespaceInfo ExpectedBWithExportedFunction(EmptySID);
  FunctionInfo ExportedF;
  ExportedF.Name = "exportedModuleFunction";
  ExportedF.ReturnType =
      TypeInfo(Reference(EmptySID, "double", InfoType::IT_default));
  ExportedF.Loc.emplace_back(0, llvm::SmallString<16>{"test.cpp"});
  ExportedF.Params.emplace_back(TypeInfo("double"), "y");
  ExportedF.Access = AccessSpecifier::AS_none;
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
  ExpectedParentA.ChildRecords.emplace_back(EmptySID, "A", InfoType::IT_record,
                                            "GlobalNamespace");
  CheckNamespaceInfo(&ExpectedParentA, ParentA);

  RecordInfo *ParentB = InfoAsRecord(Infos[3].get());
  RecordInfo ExpectedParentB(EmptySID);
  llvm::SmallString<128> ExpectedParentBPath("GlobalNamespace/A");
  llvm::sys::path::native(ExpectedParentBPath);
  ExpectedParentB.ChildRecords.emplace_back(EmptySID, "B", InfoType::IT_record,
                                            ExpectedParentBPath);
  CheckRecordInfo(&ExpectedParentB, ParentB);

  NamespaceInfo *ParentC = InfoAsNamespace(Infos[7].get());
  NamespaceInfo ExpectedParentC(EmptySID);
  ExpectedParentC.ChildRecords.emplace_back(EmptySID, "C", InfoType::IT_record,
                                            "@nonymous_namespace");
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
                                               InfoType::IT_namespace, "A");
  CheckNamespaceInfo(&ExpectedParentB, ParentB);
}

} // namespace doc
} // end namespace clang
