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
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

template <typename T> static std::string writeInfo(T &I) {
  SmallString<2048> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ClangDocBitcodeWriter Writer(Stream);
  Writer.emitBlock(I);
  return Buffer.str().str();
}

std::string writeInfo(Info *I) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    return writeInfo(*static_cast<NamespaceInfo *>(I));
  case InfoType::IT_record:
    return writeInfo(*static_cast<RecordInfo *>(I));
  case InfoType::IT_enum:
    return writeInfo(*static_cast<EnumInfo *>(I));
  case InfoType::IT_function:
    return writeInfo(*static_cast<FunctionInfo *>(I));
  case InfoType::IT_typedef:
    return writeInfo(*static_cast<TypedefInfo *>(I));
  default:
    return "";
  }
}

std::vector<std::unique_ptr<Info>> readInfo(StringRef Bitcode,
                                            size_t NumInfos) {
  llvm::BitstreamCursor Stream(Bitcode);
  doc::ClangDocBitcodeReader Reader(Stream);
  auto Infos = Reader.readBitcode();

  // Check that there was no error in the read.
  assert(Infos);
  EXPECT_EQ(Infos.get().size(), NumInfos);
  return std::move(Infos.get());
}

TEST(BitcodeTest, emitNamespaceInfoBitcode) {
  NamespaceInfo I;
  I.Name = "r";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.Children.Namespaces.emplace_back(EmptySID, "ChildNamespace",
                                     InfoType::IT_namespace);
  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record);
  I.Children.Functions.emplace_back();
  I.Children.Enums.emplace_back();

  std::string WriteResult = writeInfo(&I);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckNamespaceInfo(&I, InfoAsNamespace(ReadResults[0].get()));
}

TEST(BitcodeTest, emitRecordInfoBitcode) {
  RecordInfo I;
  I.Name = "r";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back(TypeInfo("int"), "X", AccessSpecifier::AS_private);
  I.TagType = TagTypeKind::TTK_Class;
  I.IsTypeDef = true;
  I.Bases.emplace_back(EmptySID, "F", "path/to/F", true,
                       AccessSpecifier::AS_public, true);
  I.Bases.back().Children.Functions.emplace_back();
  I.Bases.back().Members.emplace_back(TypeInfo("int"), "X",
                                      AccessSpecifier::AS_private);
  I.Parents.emplace_back(EmptySID, "F", InfoType::IT_record);
  I.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);

  // Documentation for the data member.
  CommentInfo TopComment;
  TopComment.Kind = "FullComment";
  TopComment.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *Brief = TopComment.Children.back().get();
  Brief->Kind = "ParagraphComment";
  Brief->Children.emplace_back(std::make_unique<CommentInfo>());
  Brief->Children.back()->Kind = "TextComment";
  Brief->Children.back()->Name = "ParagraphComment";
  Brief->Children.back()->Text = "Value of the thing.";
  I.Bases.back().Members.back().Description.emplace_back(std::move(TopComment));

  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record);
  I.Children.Functions.emplace_back();
  I.Children.Enums.emplace_back();

  std::string WriteResult = writeInfo(&I);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckRecordInfo(&I, InfoAsRecord(ReadResults[0].get()));
}

TEST(BitcodeTest, emitFunctionInfoBitcode) {
  FunctionInfo I;
  I.Name = "f";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.ReturnType = TypeInfo("void");
  I.Params.emplace_back(TypeInfo("int"), "P");

  I.Access = AccessSpecifier::AS_none;

  std::string WriteResult = writeInfo(&I);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckFunctionInfo(&I, InfoAsFunction(ReadResults[0].get()));
}

TEST(BitcodeTest, emitMethodInfoBitcode) {
  FunctionInfo I;
  I.Name = "f";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.ReturnType = TypeInfo("void");
  I.Params.emplace_back(TypeInfo("int"), "P");
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  I.Access = AccessSpecifier::AS_public;

  std::string WriteResult = writeInfo(&I);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckFunctionInfo(&I, InfoAsFunction(ReadResults[0].get()));
}

TEST(BitcodeTest, emitEnumInfoBitcode) {
  EnumInfo I;
  I.Name = "e";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back("X");
  I.Scoped = true;

  std::string WriteResult = writeInfo(&I);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckEnumInfo(&I, InfoAsEnum(ReadResults[0].get()));
}

TEST(BitcodeTest, emitTypedefInfoBitcode) {
  TypedefInfo I;
  I.Name = "MyInt";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Underlying = TypeInfo("unsigned");
  I.IsUsing = true;

  std::string WriteResult = writeInfo(&I);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckTypedefInfo(&I, InfoAsTypedef(ReadResults[0].get()));
}

TEST(SerializeTest, emitInfoWithCommentBitcode) {
  FunctionInfo F;
  F.Name = "F";
  F.ReturnType = TypeInfo("void");
  F.DefLoc = Location(0, llvm::SmallString<16>{"test.cpp"});
  F.Params.emplace_back(TypeInfo("int"), "I");

  CommentInfo Top;
  Top.Kind = "FullComment";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *BlankLine = Top.Children.back().get();
  BlankLine->Kind = "ParagraphComment";
  BlankLine->Children.emplace_back(std::make_unique<CommentInfo>());
  BlankLine->Children.back()->Kind = "TextComment";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *Brief = Top.Children.back().get();
  Brief->Kind = "ParagraphComment";
  Brief->Children.emplace_back(std::make_unique<CommentInfo>());
  Brief->Children.back()->Kind = "TextComment";
  Brief->Children.back()->Name = "ParagraphComment";
  Brief->Children.back()->Text = " Brief description.";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *Extended = Top.Children.back().get();
  Extended->Kind = "ParagraphComment";
  Extended->Children.emplace_back(std::make_unique<CommentInfo>());
  Extended->Children.back()->Kind = "TextComment";
  Extended->Children.back()->Text = " Extended description that";
  Extended->Children.emplace_back(std::make_unique<CommentInfo>());
  Extended->Children.back()->Kind = "TextComment";
  Extended->Children.back()->Text = " continues onto the next line.";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *HTML = Top.Children.back().get();
  HTML->Kind = "ParagraphComment";
  HTML->Children.emplace_back(std::make_unique<CommentInfo>());
  HTML->Children.back()->Kind = "TextComment";
  HTML->Children.emplace_back(std::make_unique<CommentInfo>());
  HTML->Children.back()->Kind = "HTMLStartTagComment";
  HTML->Children.back()->Name = "ul";
  HTML->Children.back()->AttrKeys.emplace_back("class");
  HTML->Children.back()->AttrValues.emplace_back("test");
  HTML->Children.emplace_back(std::make_unique<CommentInfo>());
  HTML->Children.back()->Kind = "HTMLStartTagComment";
  HTML->Children.back()->Name = "li";
  HTML->Children.emplace_back(std::make_unique<CommentInfo>());
  HTML->Children.back()->Kind = "TextComment";
  HTML->Children.back()->Text = " Testing.";
  HTML->Children.emplace_back(std::make_unique<CommentInfo>());
  HTML->Children.back()->Kind = "HTMLEndTagComment";
  HTML->Children.back()->Name = "ul";
  HTML->Children.back()->SelfClosing = true;

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *Verbatim = Top.Children.back().get();
  Verbatim->Kind = "VerbatimBlockComment";
  Verbatim->Name = "verbatim";
  Verbatim->CloseName = "endverbatim";
  Verbatim->Children.emplace_back(std::make_unique<CommentInfo>());
  Verbatim->Children.back()->Kind = "VerbatimBlockLineComment";
  Verbatim->Children.back()->Text = " The description continues.";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *ParamOut = Top.Children.back().get();
  ParamOut->Kind = "ParamCommandComment";
  ParamOut->Direction = "[out]";
  ParamOut->ParamName = "I";
  ParamOut->Explicit = true;
  ParamOut->Children.emplace_back(std::make_unique<CommentInfo>());
  ParamOut->Children.back()->Kind = "ParagraphComment";
  ParamOut->Children.back()->Children.emplace_back(
      std::make_unique<CommentInfo>());
  ParamOut->Children.back()->Children.back()->Kind = "TextComment";
  ParamOut->Children.back()->Children.emplace_back(
      std::make_unique<CommentInfo>());
  ParamOut->Children.back()->Children.back()->Kind = "TextComment";
  ParamOut->Children.back()->Children.back()->Text = " is a parameter.";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *ParamIn = Top.Children.back().get();
  ParamIn->Kind = "ParamCommandComment";
  ParamIn->Direction = "[in]";
  ParamIn->ParamName = "J";
  ParamIn->Children.emplace_back(std::make_unique<CommentInfo>());
  ParamIn->Children.back()->Kind = "ParagraphComment";
  ParamIn->Children.back()->Children.emplace_back(
      std::make_unique<CommentInfo>());
  ParamIn->Children.back()->Children.back()->Kind = "TextComment";
  ParamIn->Children.back()->Children.back()->Text = " is a parameter.";
  ParamIn->Children.back()->Children.emplace_back(
      std::make_unique<CommentInfo>());
  ParamIn->Children.back()->Children.back()->Kind = "TextComment";

  Top.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *Return = Top.Children.back().get();
  Return->Kind = "BlockCommandComment";
  Return->Name = "return";
  Return->Explicit = true;
  Return->Children.emplace_back(std::make_unique<CommentInfo>());
  Return->Children.back()->Kind = "ParagraphComment";
  Return->Children.back()->Children.emplace_back(
      std::make_unique<CommentInfo>());
  Return->Children.back()->Children.back()->Kind = "TextComment";
  Return->Children.back()->Children.back()->Text = "void";

  F.Description.emplace_back(std::move(Top));

  std::string WriteResult = writeInfo(&F);
  EXPECT_TRUE(WriteResult.size() > 0);
  std::vector<std::unique_ptr<Info>> ReadResults = readInfo(WriteResult, 1);

  CheckFunctionInfo(&F, InfoAsFunction(ReadResults[0].get()));
}

} // namespace doc
} // namespace clang
