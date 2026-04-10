//===-- clang-doc/MDGeneratorTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocTest.h"
#include "Generators.h"
#include "Representation.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

static std::unique_ptr<Generator> getMDGenerator() {
  auto G = doc::findGeneratorByName("md");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

class MDGeneratorTest : public ClangDocContextTest {};

TEST_F(MDGeneratorTest, emitNamespaceMD) {
  NamespaceInfo I;
  I.Name = "Namespace";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  Reference NewNamespace(EmptySID, "ChildNamespace", InfoType::IT_namespace);
  I.Children.Namespaces.push_back(NewNamespace);
  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record);
  I.Children.Records.push_back(ChildStruct);

  FunctionInfo F;
  F.Name = "OneFunction";
  F.Access = AccessSpecifier::AS_none;
  I.Children.Functions.push_back(F);

  EnumInfo E;
  E.Name = "OneEnum";
  I.Children.Enums.push_back(E);

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected = R"raw(# namespace Namespace



## Namespaces

* [ChildNamespace](../ChildNamespace/index.md)


## Records

* [ChildStruct](../ChildStruct.md)


## Functions

### OneFunction

* OneFunction()*



## Enums

| enum OneEnum |

| Name | Value |


)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(MDGeneratorTest, emitRecordMD) {
  RecordInfo I;
  I.Name = "r";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  MemberTypeInfo M[] = {
      MemberTypeInfo(TypeInfo("int"), "X", AccessSpecifier::AS_private)};
  I.Members = llvm::ArrayRef(M);
  I.TagType = TagTypeKind::Class;
  Reference Parents[] = {Reference(EmptySID, "F", InfoType::IT_record)};
  I.Parents = llvm::ArrayRef(Parents);
  Reference VParents[] = {Reference(EmptySID, "G", InfoType::IT_record)};
  I.VirtualParents = llvm::ArrayRef(VParents);

  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record);
  I.Children.Records.push_back(ChildStruct);
  FunctionInfo F;
  F.Name = "OneFunction";
  I.Children.Functions.push_back(F);

  EnumInfo E;
  E.Name = "OneEnum";
  I.Children.Enums.push_back(E);

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected = R"raw(# class r

*Defined at test.cpp#10*

Inherits from F, G



## Members

private int X



## Records

ChildStruct



## Functions

### OneFunction

*public  OneFunction()*



## Enums

| enum OneEnum |

| Name | Value |


)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(MDGeneratorTest, emitFunctionMD) {
  FunctionInfo I;
  I.Name = "f";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  I.Access = AccessSpecifier::AS_none;

  I.ReturnType = TypeInfo("void");
  FieldTypeInfo P[] = {FieldTypeInfo(TypeInfo("int"), "P")};
  I.Params = llvm::ArrayRef(P);
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected = R"raw(### f

*void f(int P)*

*Defined at test.cpp#10*

)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(MDGeneratorTest, emitEnumMD) {
  EnumInfo I;
  I.Name = "e";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  EnumValueInfo EV[] = {EnumValueInfo("X")};
  I.Members = llvm::ArrayRef(EV);
  I.Scoped = true;

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected = R"raw(| enum class e |

| Name | Value |
|---|---|
| X | 0 |

*Defined at test.cpp#10*

)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(MDGeneratorTest, emitCommentMD) {
  FunctionInfo I;
  I.Name = "f";

  I.DefLoc = Location(10, 10, "test.cpp");
  I.ReturnType = TypeInfo("void");
  FieldTypeInfo PI[] = {FieldTypeInfo(TypeInfo("int"), "I"),
                        FieldTypeInfo(TypeInfo("int"), "J")};
  I.Params = llvm::ArrayRef(PI);
  I.Access = AccessSpecifier::AS_none;

  CommentInfo Top;
  Top.Kind = CommentKind::CK_FullComment;

  llvm::SmallVector<CommentInfo, 8> TopChildren;

  // BlankLine
  CommentInfo BlankLine;
  BlankLine.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo BlankText;
  BlankText.Kind = CommentKind::CK_TextComment;
  CommentInfo BlankChildren[] = {std::move(BlankText)};
  BlankLine.Children = BlankChildren;
  TopChildren.push_back(std::move(BlankLine));

  // Brief
  CommentInfo Brief;
  Brief.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo BriefText;
  BriefText.Kind = CommentKind::CK_TextComment;
  BriefText.Name = "ParagraphComment";
  BriefText.Text = " Brief description.";
  CommentInfo BriefChildren[] = {std::move(BriefText)};
  Brief.Children = BriefChildren;
  TopChildren.push_back(std::move(Brief));

  // Extended
  CommentInfo Extended;
  Extended.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo ExtText1;
  ExtText1.Kind = CommentKind::CK_TextComment;
  ExtText1.Text = " Extended description that";
  CommentInfo ExtText2;
  ExtText2.Kind = CommentKind::CK_TextComment;
  ExtText2.Text = " continues onto the next line.";
  CommentInfo ExtChildren[] = {std::move(ExtText1), std::move(ExtText2)};
  Extended.Children = ExtChildren;
  TopChildren.push_back(std::move(Extended));

  // HTML
  CommentInfo HTML;
  HTML.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo HtmlText1;
  HtmlText1.Kind = CommentKind::CK_TextComment;
  CommentInfo HtmlStart;
  HtmlStart.Kind = CommentKind::CK_HTMLStartTagComment;
  HtmlStart.Name = "ul";
  StringRef Keys[] = {"class"};
  StringRef Values[] = {"test"};
  HtmlStart.AttrKeys = Keys;
  HtmlStart.AttrValues = Values;
  CommentInfo HtmlStartLi;
  HtmlStartLi.Kind = CommentKind::CK_HTMLStartTagComment;
  HtmlStartLi.Name = "li";
  CommentInfo HtmlText2;
  HtmlText2.Kind = CommentKind::CK_TextComment;
  HtmlText2.Text = " Testing.";
  CommentInfo HtmlEnd;
  HtmlEnd.Kind = CommentKind::CK_HTMLEndTagComment;
  HtmlEnd.Name = "ul";
  HtmlEnd.SelfClosing = true;
  CommentInfo HtmlChildren[] = {std::move(HtmlText1), std::move(HtmlStart),
                                std::move(HtmlStartLi), std::move(HtmlText2),
                                std::move(HtmlEnd)};
  HTML.Children = HtmlChildren;
  TopChildren.push_back(std::move(HTML));

  // Verbatim
  CommentInfo Verbatim;
  Verbatim.Kind = CommentKind::CK_VerbatimBlockComment;
  Verbatim.Name = "verbatim";
  Verbatim.CloseName = "endverbatim";
  CommentInfo VerbLine;
  VerbLine.Kind = CommentKind::CK_VerbatimBlockLineComment;
  VerbLine.Text = " The description continues.";
  CommentInfo VerbChildren[] = {std::move(VerbLine)};
  Verbatim.Children = VerbChildren;
  TopChildren.push_back(std::move(Verbatim));

  // ParamOut
  CommentInfo ParamOut;
  ParamOut.Kind = CommentKind::CK_ParamCommandComment;
  ParamOut.Direction = "[out]";
  ParamOut.ParamName = "I";
  ParamOut.Explicit = true;
  CommentInfo ParamOutPara;
  ParamOutPara.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo ParamOutText1;
  ParamOutText1.Kind = CommentKind::CK_TextComment;
  CommentInfo ParamOutText2;
  ParamOutText2.Kind = CommentKind::CK_TextComment;
  ParamOutText2.Text = "is a parameter.";
  CommentInfo ParamOutParaChildren[] = {std::move(ParamOutText1),
                                        std::move(ParamOutText2)};
  ParamOutPara.Children = ParamOutParaChildren;
  CommentInfo ParamOutChildren[] = {std::move(ParamOutPara)};
  ParamOut.Children = ParamOutChildren;
  TopChildren.push_back(std::move(ParamOut));

  // ParamIn
  CommentInfo ParamIn;
  ParamIn.Kind = CommentKind::CK_ParamCommandComment;
  ParamIn.Direction = "[in]";
  ParamIn.ParamName = "J";
  CommentInfo ParamInPara;
  ParamInPara.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo ParamInText1;
  ParamInText1.Kind = CommentKind::CK_TextComment;
  ParamInText1.Text = "is a parameter.";
  CommentInfo ParamInText2;
  ParamInText2.Kind = CommentKind::CK_TextComment;
  CommentInfo ParamInParaChildren[] = {std::move(ParamInText1),
                                       std::move(ParamInText2)};
  ParamInPara.Children = ParamInParaChildren;
  CommentInfo ParamInChildren[] = {std::move(ParamInPara)};
  ParamIn.Children = ParamInChildren;
  TopChildren.push_back(std::move(ParamIn));

  // Return
  CommentInfo Return;
  Return.Kind = CommentKind::CK_BlockCommandComment;
  Return.Name = "return";
  Return.Explicit = true;
  CommentInfo ReturnPara;
  ReturnPara.Kind = CommentKind::CK_ParagraphComment;
  CommentInfo ReturnText1;
  ReturnText1.Kind = CommentKind::CK_TextComment;
  ReturnText1.Text = "void";
  CommentInfo ReturnParaChildren[] = {std::move(ReturnText1)};
  ReturnPara.Children = ReturnParaChildren;
  CommentInfo ReturnChildren[] = {std::move(ReturnPara)};
  Return.Children = ReturnChildren;
  TopChildren.push_back(std::move(Return));

  Top.Children = TopChildren;

  I.Description.push_back(Top);

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(### f

*void f(int I, int J)*

*Defined at test.cpp#10*



 Brief description.

 Extended description that continues onto the next line.

<ul "class=test">

<li>

 Testing.</ul>



 The description continues.

**I** [out] is a parameter.

**J** is a parameter.

**return** void

)raw";

  EXPECT_EQ(Expected, Actual.str());
}

} // namespace doc
} // namespace clang
