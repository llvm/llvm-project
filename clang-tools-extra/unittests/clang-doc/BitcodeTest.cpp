//===-- clang-doc/BitcodeTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitcodeReader.h"
#include "BitcodeWriter.h"
#include "ClangDocTest.h"
#include "Representation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

template <typename T>
static std::string writeInfo(T &I, DiagnosticsEngine &Diags) {
  SmallString<2048> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ClangDocBitcodeWriter Writer(Stream, Diags);
  Writer.emitBlock(I);
  return Buffer.str().str();
}

static std::string writeInfo(Info *I, DiagnosticsEngine &Diags) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    return writeInfo(*static_cast<NamespaceInfo *>(I), Diags);
  case InfoType::IT_record:
    return writeInfo(*static_cast<RecordInfo *>(I), Diags);
  case InfoType::IT_enum:
    return writeInfo(*static_cast<EnumInfo *>(I), Diags);
  case InfoType::IT_function:
    return writeInfo(*static_cast<FunctionInfo *>(I), Diags);
  case InfoType::IT_typedef:
    return writeInfo(*static_cast<TypedefInfo *>(I), Diags);
  case InfoType::IT_concept:
    return writeInfo(*static_cast<ConceptInfo *>(I), Diags);
  case InfoType::IT_variable:
    return writeInfo(*static_cast<VarInfo *>(I), Diags);
  case InfoType::IT_friend:
    return writeInfo(*static_cast<FriendInfo *>(I), Diags);
  case InfoType::IT_default:
    return "";
  }
}

static OwningPtrVec<Info> readInfo(StringRef Bitcode, size_t NumInfos,
                                   DiagnosticsEngine &Diags) {
  llvm::BitstreamCursor Stream(Bitcode);
  doc::ClangDocBitcodeReader Reader(Stream, Diags);
  auto Infos = Reader.readBitcode();
  // Check that there was no error in the read.
  assert(Infos);
  EXPECT_EQ(Infos.get().size(), NumInfos);
  return std::move(Infos.get());
}

class BitcodeTest : public ClangDocContextTest {};

TEST_F(BitcodeTest, emitNamespaceInfoBitcode) {
  NamespaceInfo I;
  I.Name = "r";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  Reference NewNamespace(EmptySID, "ChildNamespace", InfoType::IT_namespace);
  I.Children.Namespaces.push_back(NewNamespace);
  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record);
  I.Children.Records.push_back(ChildStruct);
  FunctionInfo FI;
  I.Children.Functions.push_back(FI);
  EnumInfo EI;
  I.Children.Enums.push_back(EI);

  std::string WriteResult = writeInfo(&I, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckNamespaceInfo(&I, InfoAsNamespace(ReadResults[0]));
}

TEST_F(BitcodeTest, emitRecordInfoBitcode) {
  RecordInfo I;
  I.Name = "r";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  MemberTypeInfo M(TypeInfo("int"), "X", AccessSpecifier::AS_private);
  I.TagType = TagTypeKind::Class;
  I.IsTypeDef = true;
  BaseRecordInfo B(EmptySID, "F", "path/to/F", true, AccessSpecifier::AS_public,
                   true);
  FunctionInfo FI;
  B.Children.Functions.push_back(FI);
  MemberTypeInfo BM(TypeInfo("int"), "X", AccessSpecifier::AS_private);

  // Documentation for the data member.
  CommentInfo BriefChildren[] = {CommentInfo(CommentKind::CK_TextComment, {},
                                             "Value of the thing.",
                                             "ParagraphComment")};
  CommentInfo TopCommentChildren[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, BriefChildren)};
  CommentInfo TopComment(CommentKind::CK_FullComment, TopCommentChildren);
  BM.Description.push_back(TopComment);
  MemberTypeInfo BMem[] = {std::move(BM)};
  B.Members = llvm::ArrayRef(BMem);
  BaseRecordInfo Bases[] = {std::move(B)};
  I.Bases = llvm::ArrayRef(Bases);

  MemberTypeInfo Mem[] = {std::move(M)};
  I.Members = llvm::ArrayRef(Mem);
  Reference Parents[] = {Reference(EmptySID, "F", InfoType::IT_record)};
  I.Parents = llvm::ArrayRef(Parents);
  Reference VParents[] = {Reference(EmptySID, "G", InfoType::IT_record)};
  I.VirtualParents = llvm::ArrayRef(VParents);

  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record);
  I.Children.Records.push_back(ChildStruct);
  FunctionInfo FI2;
  I.Children.Functions.push_back(FI2);
  EnumInfo EI;
  I.Children.Enums.push_back(EI);

  std::string WriteResult = writeInfo(&I, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckRecordInfo(&I, InfoAsRecord(ReadResults[0]));
}

TEST_F(BitcodeTest, emitFunctionInfoBitcode) {
  FunctionInfo I;
  I.Name = "f";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  I.ReturnType = TypeInfo("void");
  FieldTypeInfo P(TypeInfo("int"), "P");
  FieldTypeInfo Params[] = {std::move(P)};
  I.Params = llvm::ArrayRef(Params);

  I.Access = AccessSpecifier::AS_none;

  std::string WriteResult = writeInfo(&I, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckFunctionInfo(&I, InfoAsFunction(ReadResults[0]));
}

TEST_F(BitcodeTest, emitMethodInfoBitcode) {
  FunctionInfo I;
  I.Name = "f";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  I.ReturnType = TypeInfo("void");
  FieldTypeInfo P(TypeInfo("int"), "P");
  FieldTypeInfo Params[] = {std::move(P)};
  I.Params = llvm::ArrayRef(Params);
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  I.Access = AccessSpecifier::AS_public;

  std::string WriteResult = writeInfo(&I, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckFunctionInfo(&I, InfoAsFunction(ReadResults[0]));
}

TEST_F(BitcodeTest, emitEnumInfoBitcode) {
  EnumInfo I;
  I.Name = "e";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  EnumValueInfo EV("X");
  EnumValueInfo Mems[] = {std::move(EV)};
  I.Members = llvm::ArrayRef(Mems);
  I.Scoped = true;

  std::string WriteResult = writeInfo(&I, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckEnumInfo(&I, InfoAsEnum(ReadResults[0]));
}

TEST_F(BitcodeTest, emitTypedefInfoBitcode) {
  TypedefInfo I;
  I.Name = "MyInt";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  I.Underlying = TypeInfo("unsigned");
  I.IsUsing = true;

  CommentInfo BlankChildren[] = {CommentInfo(CommentKind::CK_TextComment)};
  CommentInfo TopChildren[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, BlankChildren)};
  CommentInfo Top(CommentKind::CK_FullComment, TopChildren);

  I.Description.push_back(Top);

  std::string WriteResult = writeInfo(&I, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckTypedefInfo(&I, InfoAsTypedef(ReadResults[0]));

  // Check one with no IsUsing set, no description, and no definition location.
  TypedefInfo I2;
  I2.Name = "SomethingElse";
  I2.IsUsing = false;
  I2.Underlying = TypeInfo("int");

  WriteResult = writeInfo(&I2, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  ReadResults = readInfo(WriteResult, 1, this->Diags);
  CheckTypedefInfo(&I2, InfoAsTypedef(ReadResults[0]));
}

TEST_F(BitcodeTest, emitInfoWithCommentBitcode) {
  FunctionInfo F;
  F.Name = "F";
  F.ReturnType = TypeInfo("void");
  F.DefLoc = Location(0, 0, "test.cpp");
  FieldTypeInfo PI[] = {FieldTypeInfo(TypeInfo("int"), "I")};
  F.Params = llvm::ArrayRef(PI);

  // BlankLine
  CommentInfo BlankChildren[] = {CommentInfo(CommentKind::CK_TextComment)};
  CommentInfo BlankLine(CommentKind::CK_ParagraphComment, BlankChildren);

  // Brief
  CommentInfo BriefChildren[] = {CommentInfo(CommentKind::CK_TextComment, {},
                                             " Brief description.",
                                             "ParagraphComment")};
  CommentInfo Brief(CommentKind::CK_ParagraphComment, BriefChildren);

  // Extended
  CommentInfo ExtChildren[] = {CommentInfo(CommentKind::CK_TextComment, {},
                                           " Extended description that"),
                               CommentInfo(CommentKind::CK_TextComment, {},
                                           " continues onto the next line.")};
  CommentInfo Extended(CommentKind::CK_ParagraphComment, ExtChildren);

  // HTML
  StringRef HtmlKeys[] = {"class"};
  StringRef HtmlValues[] = {"test"};
  CommentInfo HtmlStart(CommentKind::CK_HTMLStartTagComment, {}, "", "ul", "",
                        "", "", false, false, HtmlKeys, HtmlValues);
  CommentInfo HtmlStartLi(CommentKind::CK_HTMLStartTagComment, {}, "", "li");
  CommentInfo HtmlEnd(CommentKind::CK_HTMLEndTagComment, {}, "", "ul", "", "",
                      "", false, true);

  CommentInfo HtmlChildren[] = {
      CommentInfo(CommentKind::CK_TextComment), HtmlStart, HtmlStartLi,
      CommentInfo(CommentKind::CK_TextComment, {}, " Testing."), HtmlEnd};
  CommentInfo HTML(CommentKind::CK_ParagraphComment, HtmlChildren);

  // Verbatim
  CommentInfo VerbLine(CommentKind::CK_VerbatimBlockLineComment, {},
                       " The description continues.");
  CommentInfo VerbChildren[] = {VerbLine};
  CommentInfo Verbatim(CommentKind::CK_VerbatimBlockComment, VerbChildren, "",
                       "verbatim", "endverbatim");

  // ParamOut
  CommentInfo ParamOutParaChildren[] = {
      CommentInfo(CommentKind::CK_TextComment),
      CommentInfo(CommentKind::CK_TextComment, {}, " is a parameter.")};
  CommentInfo ParamOutPara(CommentKind::CK_ParagraphComment,
                           ParamOutParaChildren);
  CommentInfo ParamOutChildren[] = {ParamOutPara};
  CommentInfo ParamOut(CommentKind::CK_ParamCommandComment, ParamOutChildren,
                       "", "", "", "[out]", "I", true);

  // ParamIn
  CommentInfo ParamInParaChildren[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, " is a parameter."),
      CommentInfo(CommentKind::CK_TextComment)};
  CommentInfo ParamInPara(CommentKind::CK_ParagraphComment,
                          ParamInParaChildren);
  CommentInfo ParamInChildren[] = {ParamInPara};
  CommentInfo ParamIn(CommentKind::CK_ParamCommandComment, ParamInChildren, "",
                      "", "", "[in]", "J");

  // Return
  CommentInfo ReturnParaChildren[] = {
      CommentInfo(CommentKind::CK_TextComment, {}, "void")};
  CommentInfo ReturnPara(CommentKind::CK_ParagraphComment, ReturnParaChildren);
  CommentInfo ReturnChildren[] = {ReturnPara};
  CommentInfo Return(CommentKind::CK_BlockCommandComment, ReturnChildren, "",
                     "return", "", "", "", true);

  CommentInfo TopChildren[] = {BlankLine, Brief,    Extended, HTML,
                               Verbatim,  ParamOut, ParamIn,  Return};
  CommentInfo Top(CommentKind::CK_FullComment, TopChildren);

  F.Description.push_back(Top);

  std::string WriteResult = writeInfo(&F, this->Diags);
  EXPECT_TRUE(WriteResult.size() > 0);
  OwningPtrVec<Info> ReadResults = readInfo(WriteResult, 1, this->Diags);

  CheckFunctionInfo(&F, InfoAsFunction(ReadResults[0]));
}

} // namespace doc
} // namespace clang
