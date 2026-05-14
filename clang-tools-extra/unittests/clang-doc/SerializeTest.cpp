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

class SerializeTest : public ClangDocContextTest {};

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
  ClangDocSerializeTestVisitor(EmittedInfoList &EmittedInfos, bool Public,
                               DiagnosticsEngine &Diags)
      : EmittedInfos(EmittedInfos), Public(Public) {}

  template <typename T> bool mapDecl(const T *D) {
    Location Loc(0, 0, "test.cpp");
    serialize::Serializer S;
    auto [Child, Parent] = S.emitInfo(D, getComment(D), Loc, Public);
    if (Child)
      EmittedInfos.emplace_back(std::move(Child));
    if (Parent)
      EmittedInfos.emplace_back(std::move(Parent));
    return true;
  }

  bool VisitNamespaceDecl(const NamespaceDecl *D) { return mapDecl(D); }

  bool VisitFunctionDecl(const FunctionDecl *D) {
    // Don't visit CXXMethodDecls twice
    if (isa<CXXMethodDecl>(D))
      return true;
    return mapDecl(D);
  }

  bool VisitCXXMethodDecl(const CXXMethodDecl *D) { return mapDecl(D); }

  bool VisitRecordDecl(const RecordDecl *D) { return mapDecl(D); }

  bool VisitEnumDecl(const EnumDecl *D) { return mapDecl(D); }

  bool VisitTypedefDecl(const TypedefDecl *D) { return mapDecl(D); }

  bool VisitTypeAliasDecl(const TypeAliasDecl *D) { return mapDecl(D); }
};

static void extractInfosFromCode(StringRef Code, size_t NumExpectedInfos,
                                 bool Public, EmittedInfoList &EmittedInfos,
                                 DiagnosticsEngine &Diags) {
  auto ASTUnit = clang::tooling::buildASTFromCode(Code);
  TranslationUnitDecl *TU = ASTUnit->getASTContext().getTranslationUnitDecl();
  ClangDocSerializeTestVisitor Visitor(EmittedInfos, Public, Diags);
  Visitor.TraverseTranslationUnitDecl(TU);
  ASSERT_EQ(NumExpectedInfos, EmittedInfos.size());
}

static void extractInfosFromCodeWithArgs(StringRef Code,
                                         size_t NumExpectedInfos, bool Public,
                                         EmittedInfoList &EmittedInfos,
                                         std::vector<std::string> &Args,
                                         DiagnosticsEngine &Diags) {
  auto ASTUnit = clang::tooling::buildASTFromCodeWithArgs(Code, Args);
  TranslationUnitDecl *TU = ASTUnit->getASTContext().getTranslationUnitDecl();
  ClangDocSerializeTestVisitor Visitor(EmittedInfos, Public, Diags);
  Visitor.TraverseTranslationUnitDecl(TU);
  ASSERT_EQ(NumExpectedInfos, EmittedInfos.size());
}

// Constructs a comment definition as the parser would for one comment line.
/* TODO uncomment this when the missing comment is fixed in emitRecordInfo and
   the code that calls this is re-enabled.
CommentInfo MakeOneLineCommentInfo(const std::string &Text) {
  CommentInfo TopComment;
  TopComment.Kind = "FullComment";
  TopComment.Children.emplace_back(allocatePtr<CommentInfo>());

  CommentInfo *Brief = TopComment.Children.back().get();
  Brief->Kind = "ParagraphComment";

  Brief->Children.emplace_back(allocatePtr<CommentInfo>());
  Brief->Children.back()->Kind = "TextComment";
  Brief->Children.back()->Name = "ParagraphComment";
  Brief->Children.back()->Text = Text;

  return TopComment;
}
*/

// Test serialization of namespace declarations.
TEST_F(SerializeTest, emitNamespaceInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("namespace A { namespace B { void f() {} } }", 5,
                       /*Public=*/false, Infos, this->Diags);

  NamespaceInfo *A = InfoAsNamespace(Infos[0]);
  NamespaceInfo ExpectedA(EmptySID, "A");
  CheckNamespaceInfo(&ExpectedA, A);

  NamespaceInfo *B = InfoAsNamespace(Infos[2]);
  NamespaceInfo ExpectedB(EmptySID, /*Name=*/"B", /*Path=*/"A");
  Reference NsB[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  ExpectedB.Namespace = llvm::ArrayRef(NsB);
  CheckNamespaceInfo(&ExpectedB, B);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[4]);
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "f";
  F.ReturnType = TypeInfo("void");
  F.DefLoc = Location(0, 0, "test.cpp");
  Reference NsF[] = {Reference(EmptySID, "B", InfoType::IT_namespace),
                     Reference(EmptySID, "A", InfoType::IT_namespace)};
  F.Namespace = llvm::ArrayRef(NsF);
  F.Access = AccessSpecifier::AS_none;
  InfoNode<FunctionInfo> FNode(&F);
  ExpectedBWithFunction.Children.Functions.push_back(FNode);
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST_F(SerializeTest, emitAnonymousNamespaceInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("namespace { }", 2, /*Public=*/false, Infos,
                       this->Diags);

  NamespaceInfo *A = InfoAsNamespace(Infos[0]);
  NamespaceInfo ExpectedA(EmptySID);
  ExpectedA.Name = "@nonymous_namespace";
  CheckNamespaceInfo(&ExpectedA, A);
}

TEST_F(SerializeTest, emitRecordInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode(R"raw(class E {
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
                       10, /*Public=*/false, Infos, this->Diags);

  RecordInfo *E = InfoAsRecord(Infos[0]);
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  Reference NsE[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedE.Namespace = llvm::ArrayRef(NsE);
  ExpectedE.TagType = TagTypeKind::Class;
  ExpectedE.DefLoc = Location(0, 0, "test.cpp");
  MemberTypeInfo MemE[] = {
      MemberTypeInfo(TypeInfo("int"), "value", AccessSpecifier::AS_public)};
  ExpectedE.Members = llvm::ArrayRef(MemE);
  // TODO the data member should have the docstring on it:
  //ExpectedE.Members.back().Description.push_back(MakeOneLineCommentInfo(" Some docs"));
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *RecordWithEConstructor = InfoAsRecord(Infos[2]);
  RecordInfo ExpectedRecordWithEConstructor(EmptySID);
  FunctionInfo EConstructor;
  EConstructor.Name = "E";
  EConstructor.Parent = Reference(EmptySID, "E", InfoType::IT_record);
  EConstructor.ReturnType = TypeInfo("void");
  EConstructor.DefLoc = Location(0, 0, "test.cpp");
  Reference NsEC[] = {
      Reference(EmptySID, "E", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  EConstructor.Namespace = llvm::ArrayRef(NsEC);
  EConstructor.Access = AccessSpecifier::AS_public;
  EConstructor.IsMethod = true;
  InfoNode<FunctionInfo> EConstructorNode(&EConstructor);
  ExpectedRecordWithEConstructor.Children.Functions.push_back(EConstructorNode);
  CheckRecordInfo(&ExpectedRecordWithEConstructor, RecordWithEConstructor);

  RecordInfo *RecordWithMethod = InfoAsRecord(Infos[3]);
  RecordInfo ExpectedRecordWithMethod(EmptySID);
  FunctionInfo Method;
  Method.Name = "ProtectedMethod";
  Method.Parent = Reference(EmptySID, "E", InfoType::IT_record);
  Method.ReturnType = TypeInfo("void");
  Location LMethod(0, 0, "test.cpp");
  InfoNode<Location> LMethodNode(&LMethod);
  Method.Loc.push_back(LMethodNode);
  Reference NsMethod[] = {
      Reference(EmptySID, "E", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  Method.Namespace = llvm::ArrayRef(NsMethod);
  Method.Access = AccessSpecifier::AS_protected;
  Method.IsMethod = true;
  InfoNode<FunctionInfo> MethodNode(&Method);
  ExpectedRecordWithMethod.Children.Functions.push_back(MethodNode);
  CheckRecordInfo(&ExpectedRecordWithMethod, RecordWithMethod);

  RecordInfo *F = InfoAsRecord(Infos[4]);
  RecordInfo ExpectedF(EmptySID, /*Name=*/"F", /*Path=*/"GlobalNamespace");
  Reference NsF3[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedF.Namespace = llvm::ArrayRef(NsF3);
  ExpectedF.TagType = TagTypeKind::Struct;
  ExpectedF.DefLoc = Location(0, 0, "test.cpp");
  CheckRecordInfo(&ExpectedF, F);

  RecordInfo *RecordWithTemplateMethod = InfoAsRecord(Infos[6]);
  RecordInfo ExpectedRecordWithTemplateMethod(EmptySID);
  FunctionInfo TemplateMethod;
  TemplateMethod.Name = "TemplateMethod";
  TemplateMethod.Parent = Reference(EmptySID, "F", InfoType::IT_record);
  TemplateMethod.ReturnType = TypeInfo("void");
  Location LTemp1(0, 0, "test.cpp");
  InfoNode<Location> LTemp1Node(&LTemp1);
  TemplateMethod.Loc.push_back(LTemp1Node);
  Reference NsT1[] = {
      Reference(EmptySID, "F", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  TemplateMethod.Namespace = llvm::ArrayRef(NsT1);
  TemplateMethod.Access = AccessSpecifier::AS_public;
  TemplateMethod.IsMethod = true;
  InfoNode<FunctionInfo> TemplateMethodNode(&TemplateMethod);
  ExpectedRecordWithTemplateMethod.Children.Functions.push_back(
      TemplateMethodNode);
  CheckRecordInfo(&ExpectedRecordWithTemplateMethod, RecordWithTemplateMethod);

  RecordInfo *TemplatedRecord = InfoAsRecord(Infos[7]);
  RecordInfo ExpectedTemplatedRecord(EmptySID);
  FunctionInfo SpecializedTemplateMethod;
  SpecializedTemplateMethod.Name = "TemplateMethod";
  SpecializedTemplateMethod.Parent =
      Reference(EmptySID, "F", InfoType::IT_record);
  SpecializedTemplateMethod.ReturnType = TypeInfo("void");
  Location LTemp2(0, 0, "test.cpp");
  InfoNode<Location> LTemp2Node(&LTemp2);
  SpecializedTemplateMethod.Loc.push_back(LTemp2Node);
  Reference NsT2[] = {
      Reference(EmptySID, "F", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  SpecializedTemplateMethod.Namespace = llvm::ArrayRef(NsT2);
  SpecializedTemplateMethod.Access = AccessSpecifier::AS_public;
  SpecializedTemplateMethod.IsMethod = true;
  InfoNode<FunctionInfo> SpecializedTemplateMethodNode(
      &SpecializedTemplateMethod);
  ExpectedTemplatedRecord.Children.Functions.push_back(
      SpecializedTemplateMethodNode);
  CheckRecordInfo(&ExpectedTemplatedRecord, TemplatedRecord);

  RecordInfo *G = InfoAsRecord(Infos[8]);
  RecordInfo ExpectedG(EmptySID, /*Name=*/"G", /*Path=*/"GlobalNamespace");
  Reference NsG[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedG.Namespace = llvm::ArrayRef(NsG);
  ExpectedG.TagType = TagTypeKind::Struct;
  ExpectedG.DefLoc = Location(0, 0, "test.cpp");
  ExpectedG.IsTypeDef = true;
  CheckRecordInfo(&ExpectedG, G);
}

// Test serialization of enum declarations.
TEST_F(SerializeTest, emitEnumInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("enum E { X, Y }; enum class G { A, B };", 2,
                       /*Public=*/false, Infos, this->Diags);

  NamespaceInfo *NamespaceWithEnum = InfoAsNamespace(Infos[0]);
  NamespaceInfo ExpectedNamespaceWithEnum(EmptySID);
  EnumInfo E;
  E.Name = "E";
  E.DefLoc = Location(0, 0, "test.cpp");
  EnumValueInfo EMem[] = {EnumValueInfo("X", "0"), EnumValueInfo("Y", "1")};
  E.Members = llvm::ArrayRef(EMem);
  InfoNode<EnumInfo> ENode(&E);
  ExpectedNamespaceWithEnum.Children.Enums.push_back(ENode);
  CheckNamespaceInfo(&ExpectedNamespaceWithEnum, NamespaceWithEnum);

  NamespaceInfo *NamespaceWithScopedEnum = InfoAsNamespace(Infos[1]);
  NamespaceInfo ExpectedNamespaceWithScopedEnum(EmptySID);
  EnumInfo G;
  G.Name = "G";
  G.Scoped = true;
  G.DefLoc = Location(0, 0, "test.cpp");
  EnumValueInfo GMem[] = {EnumValueInfo("A", "0"), EnumValueInfo("B", "1")};
  G.Members = llvm::ArrayRef(GMem);
  InfoNode<EnumInfo> GNode(&G);
  ExpectedNamespaceWithScopedEnum.Children.Enums.push_back(GNode);
  CheckNamespaceInfo(&ExpectedNamespaceWithScopedEnum, NamespaceWithScopedEnum);
}

TEST_F(SerializeTest, emitUndefinedRecordInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("class E;", 2, /*Public=*/false, Infos, this->Diags);

  RecordInfo *E = InfoAsRecord(Infos[0]);
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  Reference NsE[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedE.Namespace = llvm::ArrayRef(NsE);
  ExpectedE.TagType = TagTypeKind::Class;
  Location LE(0, 0, "test.cpp");
  InfoNode<Location> LENode(&LE);
  ExpectedE.Loc.push_back(LENode);
  CheckRecordInfo(&ExpectedE, E);
}

TEST_F(SerializeTest, emitRecordMemberInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("struct E { int I; };", 2, /*Public=*/false, Infos,
                       this->Diags);

  RecordInfo *E = InfoAsRecord(Infos[0]);
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  Reference NsE[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedE.Namespace = llvm::ArrayRef(NsE);
  ExpectedE.TagType = TagTypeKind::Struct;
  ExpectedE.DefLoc = Location(0, 0, "test.cpp");
  MemberTypeInfo MemE[] = {
      MemberTypeInfo(TypeInfo("int"), "I", AccessSpecifier::AS_public)};
  ExpectedE.Members = llvm::ArrayRef(MemE);
  CheckRecordInfo(&ExpectedE, E);
}

TEST_F(SerializeTest, emitInternalRecordInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("class E { class G {}; };", 4, /*Public=*/false, Infos,
                       this->Diags);

  RecordInfo *E = InfoAsRecord(Infos[0]);
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  Reference NsE[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedE.Namespace = llvm::ArrayRef(NsE);
  ExpectedE.DefLoc = Location(0, 0, "test.cpp");
  ExpectedE.TagType = TagTypeKind::Class;
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *G = InfoAsRecord(Infos[2]);
  llvm::SmallString<128> ExpectedGPath("GlobalNamespace/E");
  llvm::sys::path::native(ExpectedGPath);
  RecordInfo ExpectedG(EmptySID, /*Name=*/"G", /*Path=*/ExpectedGPath);
  ExpectedG.DefLoc = Location(0, 0, "test.cpp");
  ExpectedG.TagType = TagTypeKind::Class;
  Reference NsG[] = {
      Reference(EmptySID, "E", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedG.Namespace = llvm::ArrayRef(NsG);
  CheckRecordInfo(&ExpectedG, G);
}

TEST_F(SerializeTest, emitPublicAnonymousNamespaceInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("namespace { class A; }", 0, /*Public=*/true, Infos,
                       this->Diags);
}

TEST_F(SerializeTest, emitPublicFunctionInternalInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("int F() { class G {}; return 0; };", 1, /*Public=*/true,
                       Infos, this->Diags);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0]);
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "F";
  F.ReturnType = TypeInfo("int");
  F.DefLoc = Location(0, 0, "test.cpp");
  F.Access = AccessSpecifier::AS_none;
  InfoNode<FunctionInfo> FNode(&F);
  ExpectedBWithFunction.Children.Functions.push_back(FNode);
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST_F(SerializeTest, emitInlinedFunctionInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode("inline void F(int I) { };", 1, /*Public=*/true, Infos,
                       this->Diags);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0]);
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "F";
  F.ReturnType = TypeInfo("void");
  F.DefLoc = Location(0, 0, "test.cpp");
  FieldTypeInfo Params[] = {FieldTypeInfo(TypeInfo("int"), "I")};
  F.Params = llvm::ArrayRef(Params);
  F.Access = AccessSpecifier::AS_none;
  InfoNode<FunctionInfo> FNode(&F);
  ExpectedBWithFunction.Children.Functions.push_back(FNode);
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);
}

TEST_F(SerializeTest, emitInheritedRecordInfo) {
  EmittedInfoList Infos;
  extractInfosFromCode(R"raw(class F { protected: void set(int N); };
class G { public: int get() { return 1; } protected: int I; };
class E : public F, virtual private G {};
class H : private E {};
template <typename T>
class I {} ;
class J : public I<int> {} ;)raw",
                       14, /*Public=*/false, Infos, this->Diags);

  RecordInfo *F = InfoAsRecord(Infos[0]);
  RecordInfo ExpectedF(EmptySID, /*Name=*/"F", /*Path=*/"GlobalNamespace");
  Reference NsF[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedF.Namespace = llvm::ArrayRef(NsF);
  ExpectedF.TagType = TagTypeKind::Class;
  ExpectedF.DefLoc = Location(0, 0, "test.cpp");
  CheckRecordInfo(&ExpectedF, F);

  RecordInfo *G = InfoAsRecord(Infos[3]);
  RecordInfo ExpectedG(EmptySID, /*Name=*/"G", /*Path=*/"GlobalNamespace");
  Reference NsG[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedG.Namespace = llvm::ArrayRef(NsG);
  ExpectedG.TagType = TagTypeKind::Class;
  ExpectedG.DefLoc = Location(0, 0, "test.cpp");
  MemberTypeInfo MemG[] = {
      MemberTypeInfo(TypeInfo("int"), "I", AccessSpecifier::AS_protected)};
  ExpectedG.Members = llvm::ArrayRef(MemG);
  CheckRecordInfo(&ExpectedG, G);

  RecordInfo *E = InfoAsRecord(Infos[6]);
  RecordInfo ExpectedE(EmptySID, /*Name=*/"E", /*Path=*/"GlobalNamespace");
  Reference NsE[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedE.Namespace = llvm::ArrayRef(NsE);
  Reference ParE[] = {Reference(EmptySID, /*Name=*/"F", InfoType::IT_record,
                                /*QualName=*/"", /*Path=*/"GlobalNamespace")};
  ExpectedE.Parents = llvm::ArrayRef(ParE);
  Reference VParE[] = {Reference(EmptySID, /*Name=*/"G", InfoType::IT_record,
                                 /*QualName=*/"G",
                                 /*Path=*/"GlobalNamespace")};
  ExpectedE.VirtualParents = llvm::ArrayRef(VParE);
  BaseRecordInfo BaseF(EmptySID, /*Name=*/"F",
                       /*Path=*/"GlobalNamespace", false,
                       AccessSpecifier::AS_public, true);
  FunctionInfo FunctionSet;
  FunctionSet.Name = "set";
  FunctionSet.ReturnType = TypeInfo("void");
  Location LSet;
  InfoNode<Location> LSetNode(&LSet);
  FunctionSet.Loc.push_back(LSetNode);
  FieldTypeInfo ParamsSet[] = {FieldTypeInfo(TypeInfo("int"), "N")};
  FunctionSet.Params = llvm::ArrayRef(ParamsSet);
  Reference NsSet[] = {
      Reference(EmptySID, "F", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  FunctionSet.Namespace = llvm::ArrayRef(NsSet);
  FunctionSet.Access =
      AccessSpecifier::AS_none; // Wait, previous had AS_protected, but wait,
                                // F.Access was AS_protected. FunctionSet.Access
                                // should be AS_protected if it was so. In the
                                // original it was AS_protected.
  FunctionSet.Access = AccessSpecifier::AS_protected;
  FunctionSet.IsMethod = true;
  InfoNode<FunctionInfo> FunctionSetNode(&FunctionSet);
  BaseF.Children.Functions.push_back(FunctionSetNode);

  BaseRecordInfo BaseG(EmptySID, /*Name=*/"G",
                       /*Path=*/"GlobalNamespace", true,
                       AccessSpecifier::AS_private, true);
  FunctionInfo FunctionGet;
  FunctionGet.Name = "get";
  FunctionGet.ReturnType = TypeInfo("int");
  Location LGet;
  FunctionGet.DefLoc = LGet;
  Reference NsGet[] = {
      Reference(EmptySID, "G", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  FunctionGet.Namespace = llvm::ArrayRef(NsGet);
  FunctionGet.Access = AccessSpecifier::AS_private;
  FunctionGet.IsMethod = true;
  InfoNode<FunctionInfo> FunctionGetNode(&FunctionGet);
  BaseG.Children.Functions.push_back(FunctionGetNode);
  MemberTypeInfo MemG2[] = {
      MemberTypeInfo(TypeInfo("int"), "I", AccessSpecifier::AS_private)};
  BaseG.Members = llvm::ArrayRef(MemG2);

  BaseRecordInfo BasesE[] = {std::move(BaseF), std::move(BaseG)};
  ExpectedE.Bases = llvm::ArrayRef(BasesE);
  ExpectedE.DefLoc = Location(0, 0, "test.cpp");
  ExpectedE.TagType = TagTypeKind::Class;
  CheckRecordInfo(&ExpectedE, E);

  RecordInfo *H = InfoAsRecord(Infos[8]);
  RecordInfo ExpectedH(EmptySID, /*Name=*/"H", /*Path=*/"GlobalNamespace");
  Reference NsH[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedH.Namespace = llvm::ArrayRef(NsH);
  ExpectedH.TagType = TagTypeKind::Class;
  ExpectedH.DefLoc = Location(0, 0, "test.cpp");
  Reference ParH[] = {Reference(EmptySID, /*Name=*/"E", InfoType::IT_record,
                                /*QualName=*/"E", /*Path=*/"GlobalNamespace")};
  ExpectedH.Parents = llvm::ArrayRef(ParH);
  Reference VParH[] = {Reference(EmptySID, /*Name=*/"G", InfoType::IT_record,
                                 /*QualName=*/"G",
                                 /*Path=*/"GlobalNamespace")};
  ExpectedH.VirtualParents = llvm::ArrayRef(VParH);

  BaseRecordInfo BaseHE(EmptySID, /*Name=*/"E",
                        /*Path=*/"GlobalNamespace", false,
                        AccessSpecifier::AS_private, true);

  BaseRecordInfo BaseHF(EmptySID, /*Name=*/"F",
                        /*Path=*/"GlobalNamespace", false,
                        AccessSpecifier::AS_private, false);
  FunctionInfo FunctionSetNew;
  FunctionSetNew.Name = "set";
  FunctionSetNew.ReturnType = TypeInfo("void");
  Location LSetNew;
  InfoNode<Location> LSetNewNode(&LSetNew);
  FunctionSetNew.Loc.push_back(LSetNewNode);
  FieldTypeInfo ParamsSetNew[] = {FieldTypeInfo(TypeInfo("int"), "N")};
  FunctionSetNew.Params = llvm::ArrayRef(ParamsSetNew);
  Reference NsSetNew[] = {
      Reference(EmptySID, "F", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  FunctionSetNew.Namespace = llvm::ArrayRef(NsSetNew);
  FunctionSetNew.Access = AccessSpecifier::AS_private;
  FunctionSetNew.IsMethod = true;
  InfoNode<FunctionInfo> FunctionSetNewNode(&FunctionSetNew);
  BaseHF.Children.Functions.push_back(FunctionSetNewNode);
  BaseRecordInfo BaseHG(EmptySID, /*Name=*/"G",
                        /*Path=*/"GlobalNamespace", true,
                        AccessSpecifier::AS_private, false);
  FunctionInfo FunctionGetNew;
  FunctionGetNew.Name = "get";
  FunctionGetNew.ReturnType = TypeInfo("int");
  Location LGetNew;
  FunctionGetNew.DefLoc = LGetNew;
  Reference NsGetNew[] = {
      Reference(EmptySID, "G", InfoType::IT_record),
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  FunctionGetNew.Namespace = llvm::ArrayRef(NsGetNew);
  FunctionGetNew.Access = AccessSpecifier::AS_private;
  FunctionGetNew.IsMethod = true;
  InfoNode<FunctionInfo> FunctionGetNewNode(&FunctionGetNew);
  BaseHG.Children.Functions.push_back(FunctionGetNewNode);
  MemberTypeInfo MemHG[] = {
      MemberTypeInfo(TypeInfo("int"), "I", AccessSpecifier::AS_private)};
  BaseHG.Members = llvm::ArrayRef(MemHG);

  BaseRecordInfo BasesH[] = {std::move(BaseHE), std::move(BaseHF),
                             std::move(BaseHG)};
  ExpectedH.Bases = llvm::ArrayRef(BasesH);

  CheckRecordInfo(&ExpectedH, H);

  RecordInfo *I = InfoAsRecord(Infos[10]);
  RecordInfo ExpectedI(EmptySID, /*Name=*/"I", /*Path=*/"GlobalNamespace");
  Reference NsI[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedI.Namespace = llvm::ArrayRef(NsI);
  ExpectedI.TagType = TagTypeKind::Class;
  ExpectedI.DefLoc = Location(0, 0, "test.cpp");
  CheckRecordInfo(&ExpectedI, I);

  RecordInfo *J = InfoAsRecord(Infos[12]);
  RecordInfo ExpectedJ(EmptySID, /*Name=*/"J", /*Path=*/"GlobalNamespace");
  Reference NsJ[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  ExpectedJ.Namespace = llvm::ArrayRef(NsJ);
  Reference ParJ[] = {
      Reference(EmptySID, /*Name=*/"I<int>", InfoType::IT_record)};
  ExpectedJ.Parents = llvm::ArrayRef(ParJ);
  BaseRecordInfo BasesJ[] = {BaseRecordInfo(EmptySID, /*Name=*/"I<int>",
                                            /*Path=*/"GlobalNamespace", false,
                                            AccessSpecifier::AS_public, true)};
  ExpectedJ.Bases = llvm::ArrayRef(BasesJ);
  ExpectedJ.DefLoc = Location(0, 0, "test.cpp");
  ExpectedJ.TagType = TagTypeKind::Class;
  CheckRecordInfo(&ExpectedJ, J);
}

TEST_F(SerializeTest, emitModulePublicLFunctions) {
  EmittedInfoList Infos;
  std::vector<std::string> Args;
  Args.push_back("-fmodules-ts");
  extractInfosFromCodeWithArgs(R"raw(export module M;
int moduleFunction(int x, double d = 3.2 - 1.0);
static int staticModuleFunction(int x);
export double exportedModuleFunction(double y);)raw",
                               2, /*Public=*/true, Infos, Args, this->Diags);

  NamespaceInfo *BWithFunction = InfoAsNamespace(Infos[0]);
  NamespaceInfo ExpectedBWithFunction(EmptySID);
  FunctionInfo F;
  F.Name = "moduleFunction";
  F.ReturnType = TypeInfo("int");
  Location LF1(0, 0, "test.cpp");
  InfoNode<Location> LF1Node(&LF1);
  F.Loc.push_back(LF1Node);
  FieldTypeInfo ParamsF[] = {FieldTypeInfo(TypeInfo("int"), "x"),
                             FieldTypeInfo(TypeInfo("double"), "d")};
  ParamsF[1].DefaultValue = "3.2 - 1.0";
  F.Params = llvm::ArrayRef(ParamsF);
  F.Access = AccessSpecifier::AS_none;
  InfoNode<FunctionInfo> FNode(&F);
  ExpectedBWithFunction.Children.Functions.push_back(FNode);
  CheckNamespaceInfo(&ExpectedBWithFunction, BWithFunction);

  NamespaceInfo *BWithExportedFunction = InfoAsNamespace(Infos[1]);
  NamespaceInfo ExpectedBWithExportedFunction(EmptySID);
  FunctionInfo ExportedF;
  ExportedF.Name = "exportedModuleFunction";
  ExportedF.ReturnType =
      TypeInfo(Reference(EmptySID, "double", InfoType::IT_default));
  Location LF2(0, 0, "test.cpp");
  InfoNode<Location> LF2Node(&LF2);
  ExportedF.Loc.push_back(LF2Node);
  FieldTypeInfo ParamsExportedF[] = {FieldTypeInfo(TypeInfo("double"), "y")};
  ExportedF.Params = llvm::ArrayRef(ParamsExportedF);
  ExportedF.Access = AccessSpecifier::AS_none;
  InfoNode<FunctionInfo> ExportedFNode(&ExportedF);
  ExpectedBWithExportedFunction.Children.Functions.push_back(ExportedFNode);
  CheckNamespaceInfo(&ExpectedBWithExportedFunction, BWithExportedFunction);
}

// Test serialization of child records in namespaces and other records
TEST_F(SerializeTest, emitChildRecords) {
  EmittedInfoList Infos;
  extractInfosFromCode("class A { class B {}; }; namespace { class C {}; } ", 8,
                       /*Public=*/false, Infos, this->Diags);

  NamespaceInfo *ParentA = InfoAsNamespace(Infos[1]);
  NamespaceInfo ExpectedParentA(EmptySID);
  Reference RA(EmptySID, "A", InfoType::IT_record, "A", "GlobalNamespace");
  InfoNode<Reference> RANode(&RA);
  ExpectedParentA.Children.Records.push_back(RANode);
  CheckNamespaceInfo(&ExpectedParentA, ParentA);

  RecordInfo *ParentB = InfoAsRecord(Infos[3]);
  RecordInfo ExpectedParentB(EmptySID);
  llvm::SmallString<128> ExpectedParentBPath("GlobalNamespace/A");
  llvm::sys::path::native(ExpectedParentBPath);
  Reference RB(EmptySID, "B", InfoType::IT_record, "A::B", ExpectedParentBPath);
  InfoNode<Reference> RBNode(&RB);
  ExpectedParentB.Children.Records.push_back(RBNode);
  CheckRecordInfo(&ExpectedParentB, ParentB);

  NamespaceInfo *ParentC = InfoAsNamespace(Infos[7]);
  NamespaceInfo ExpectedParentC(EmptySID);
  Reference RC(EmptySID, "C", InfoType::IT_record, "C", "@nonymous_namespace");
  InfoNode<Reference> RCNode(&RC);
  ExpectedParentC.Children.Records.push_back(RCNode);
  CheckNamespaceInfo(&ExpectedParentC, ParentC);
}

// Test serialization of child namespaces
TEST_F(SerializeTest, emitChildNamespaces) {
  EmittedInfoList Infos;
  extractInfosFromCode("namespace A { namespace B { } }", 4, /*Public=*/false,
                       Infos, this->Diags);

  NamespaceInfo *ParentA = InfoAsNamespace(Infos[1]);
  NamespaceInfo ExpectedParentA(EmptySID);
  Reference RA(EmptySID, "A", InfoType::IT_namespace);
  InfoNode<Reference> RANode(&RA);
  ExpectedParentA.Children.Namespaces.push_back(RANode);
  CheckNamespaceInfo(&ExpectedParentA, ParentA);

  NamespaceInfo *ParentB = InfoAsNamespace(Infos[3]);
  NamespaceInfo ExpectedParentB(EmptySID);
  Reference RB(EmptySID, "B", InfoType::IT_namespace, "A::B", "A");
  InfoNode<Reference> RBNode(&RB);
  ExpectedParentB.Children.Namespaces.push_back(RBNode);
  CheckNamespaceInfo(&ExpectedParentB, ParentB);
}

TEST_F(SerializeTest, emitTypedefs) {
  EmittedInfoList Infos;
  extractInfosFromCode("typedef int MyInt; using MyDouble = double;", 2,
                       /*Public=*/false, Infos, this->Diags);

  // First info will be the global namespace with the typedef in it.
  NamespaceInfo *GlobalNS1 = InfoAsNamespace(Infos[0]);
  ASSERT_EQ(1u, GlobalNS1->Children.Typedefs.size());

  const TypedefInfo &FirstTD = *GlobalNS1->Children.Typedefs.begin();
  EXPECT_EQ("MyInt", FirstTD.Name);
  EXPECT_FALSE(FirstTD.IsUsing);
  EXPECT_EQ("int", FirstTD.Underlying.Type.Name);

  // The second will be another global namespace with the using in it (the
  // global namespace is duplicated because the items haven't been merged at the
  // serialization phase of processing).
  NamespaceInfo *GlobalNS2 = InfoAsNamespace(Infos[1]);
  ASSERT_EQ(1u, GlobalNS2->Children.Typedefs.size());

  // Second is the "using" typedef.
  const TypedefInfo &SecondTD = *GlobalNS2->Children.Typedefs.begin();
  EXPECT_EQ("MyDouble", SecondTD.Name);
  EXPECT_TRUE(SecondTD.IsUsing);
  EXPECT_EQ("double", SecondTD.Underlying.Type.Name);
}

TEST_F(SerializeTest, emitFunctionTemplate) {
  EmittedInfoList Infos;
  // A template and a specialization.
  extractInfosFromCode("template<typename T = int> bool GetFoo(T);\n"
                       "template<> bool GetFoo<bool>(bool);",
                       2,
                       /*Public=*/false, Infos, this->Diags);

  // First info will be the global namespace.
  NamespaceInfo *GlobalNS1 = InfoAsNamespace(Infos[0]);
  ASSERT_EQ(1u, GlobalNS1->Children.Functions.size());

  const FunctionInfo &Func1 = *GlobalNS1->Children.Functions.begin();
  EXPECT_EQ("GetFoo", Func1.Name);
  ASSERT_TRUE(Func1.Template);
  EXPECT_FALSE(Func1.Template->Specialization); // Not a specialization.

  // Template parameter.
  ASSERT_EQ(1u, Func1.Template->Params.size());
  EXPECT_EQ("typename T = int", Func1.Template->Params[0].Contents);

  // The second will be another global namespace with the function in it (the
  // global namespace is duplicated because the items haven't been merged at the
  // serialization phase of processing).
  NamespaceInfo *GlobalNS2 = InfoAsNamespace(Infos[1]);
  ASSERT_EQ(1u, GlobalNS2->Children.Functions.size());

  // This one is a template specialization.
  const FunctionInfo &Func2 = *GlobalNS2->Children.Functions.begin();
  EXPECT_EQ("GetFoo", Func2.Name);
  ASSERT_TRUE(Func2.Template);
  EXPECT_TRUE(Func2.Template->Params.empty()); // No template params.
  ASSERT_TRUE(Func2.Template->Specialization);

  // Specialization values.
  ASSERT_EQ(1u, Func2.Template->Specialization->Params.size());
  EXPECT_EQ("bool", Func2.Template->Specialization->Params[0].Contents);
  EXPECT_EQ(Func1.USR, Func2.Template->Specialization->SpecializationOf);

  EXPECT_EQ("bool", Func2.ReturnType.Type.Name);
}

TEST_F(SerializeTest, emitClassTemplate) {
  EmittedInfoList Infos;
  // This will generate 2x the number of infos: each Record will be followed by
  // a copy of the global namespace containing it (this test checks the data
  // pre-merge).
  extractInfosFromCode(
      "template<int I> class MyTemplate { int i[I]; };\n"
      "template<> class MyTemplate<0> {};\n"
      "template<typename T, int U = 1> class OtherTemplate {};\n"
      "template<int U> class OtherTemplate<MyTemplate<0>, U> {};",
      8,
      /*Public=*/false, Infos, this->Diags);

  // First record.
  const RecordInfo *Rec1 = InfoAsRecord(Infos[0]);
  EXPECT_EQ("MyTemplate", Rec1->Name);
  ASSERT_TRUE(Rec1->Template);
  EXPECT_FALSE(Rec1->Template->Specialization); // Not a specialization.

  // First record template parameter.
  ASSERT_EQ(1u, Rec1->Template->Params.size());
  EXPECT_EQ("int I", Rec1->Template->Params[0].Contents);

  // Second record.
  const RecordInfo *Rec2 = InfoAsRecord(Infos[2]);
  EXPECT_EQ("MyTemplate", Rec2->Name);
  ASSERT_TRUE(Rec2->Template);
  EXPECT_TRUE(Rec2->Template->Params.empty()); // No template params.
  ASSERT_TRUE(Rec2->Template->Specialization);

  // Second record specialization values.
  ASSERT_EQ(1u, Rec2->Template->Specialization->Params.size());
  EXPECT_EQ("0", Rec2->Template->Specialization->Params[0].Contents);
  EXPECT_EQ(Rec1->USR, Rec2->Template->Specialization->SpecializationOf);

  // Third record.
  const RecordInfo *Rec3 = InfoAsRecord(Infos[4]);
  EXPECT_EQ("OtherTemplate", Rec3->Name);
  ASSERT_TRUE(Rec3->Template);

  // Third record template parameters.
  ASSERT_EQ(2u, Rec3->Template->Params.size());
  EXPECT_EQ("typename T", Rec3->Template->Params[0].Contents);
  EXPECT_EQ("int U = 1", Rec3->Template->Params[1].Contents);

  // Fourth record.
  const RecordInfo *Rec4 = InfoAsRecord(Infos[6]);
  EXPECT_EQ("OtherTemplate", Rec3->Name);
  ASSERT_TRUE(Rec4->Template);
  ASSERT_TRUE(Rec4->Template->Specialization);

  // Fourth record template + specialization parameters.
  ASSERT_EQ(1u, Rec4->Template->Params.size());
  EXPECT_EQ("int U", Rec4->Template->Params[0].Contents);
  ASSERT_EQ(2u, Rec4->Template->Specialization->Params.size());
  EXPECT_EQ("MyTemplate<0>",
            Rec4->Template->Specialization->Params[0].Contents);
  EXPECT_EQ("U", Rec4->Template->Specialization->Params[1].Contents);
}

} // namespace doc
} // end namespace clang
