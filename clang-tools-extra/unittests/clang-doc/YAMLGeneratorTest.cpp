//===-- clang-doc/YAMLGeneratorTest.cpp
//------------------------------------===//
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

static std::unique_ptr<Generator> getYAMLGenerator() {
  auto G = doc::findGeneratorByName("yaml");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

class YAMLGeneratorTest : public ClangDocContextTest {};

TEST_F(YAMLGeneratorTest, emitNamespaceYAML) {
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Path = "path/to/A";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  Reference NewNamespace(EmptySID, "ChildNamespace", InfoType::IT_namespace,
                         "path::to::A::Namespace::ChildNamespace",
                         "path/to/A/Namespace");
  I.Children.Namespaces.push_back(NewNamespace);
  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record,
                        "path::to::A::Namespace::ChildStruct",
                        "path/to/A/Namespace");
  I.Children.Records.push_back(ChildStruct);
  FunctionInfo F;
  F.Name = "OneFunction";
  F.Access = AccessSpecifier::AS_none;
  I.Children.Functions.push_back(F);

  EnumInfo E;
  E.Name = "OneEnum";
  I.Children.Enums.push_back(E);

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'Namespace'
Path:            'path/to/A'
Namespace:
  - Type:            Namespace
    Name:            'A'
    QualName:        'A'
ChildNamespaces:
  - Type:            Namespace
    Name:            'ChildNamespace'
    QualName:        'path::to::A::Namespace::ChildNamespace'
    Path:            'path/to/A/Namespace'
ChildRecords:
  - Type:            Record
    Name:            'ChildStruct'
    QualName:        'path::to::A::Namespace::ChildStruct'
    Path:            'path/to/A/Namespace'
ChildFunctions:
  - USR:             '0000000000000000000000000000000000000000'
    Name:            'OneFunction'
    ReturnType:      {}
ChildEnums:
  - USR:             '0000000000000000000000000000000000000000'
    Name:            'OneEnum'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(YAMLGeneratorTest, emitRecordYAML) {
  RecordInfo I;
  I.Name = "r";
  I.Path = "path/to/A";
  I.IsTypeDef = true;
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  MemberTypeInfo M(TypeInfo("int"), "X", AccessSpecifier::AS_private);

  // Member documentation.
  CommentInfo BriefChildren[] = {CommentInfo(CommentKind::CK_TextComment, {},
                                             "Value of the thing.",
                                             "ParagraphComment")};
  CommentInfo TopCommentChildren[] = {
      CommentInfo(CommentKind::CK_ParagraphComment, BriefChildren)};
  CommentInfo TopComment(CommentKind::CK_FullComment, TopCommentChildren);
  M.Description.push_back(TopComment);
  MemberTypeInfo MemArr[] = {std::move(M)};
  I.Members = llvm::ArrayRef(MemArr);

  I.TagType = TagTypeKind::Class;
  BaseRecordInfo B(EmptySID, "F", "path/to/F", true, AccessSpecifier::AS_public,
                   true);
  FunctionInfo F;
  F.Name = "InheritedFunctionOne";
  B.Children.Functions.push_back(F);
  MemberTypeInfo BMem[] = {
      MemberTypeInfo(TypeInfo("int"), "N", AccessSpecifier::AS_private)};
  B.Members = llvm::ArrayRef(BMem);
  BaseRecordInfo Bases[] = {std::move(B)};
  I.Bases = llvm::ArrayRef(Bases);

  // F is in the global namespace
  Reference Parents[] = {Reference(EmptySID, "F", InfoType::IT_record, "")};
  I.Parents = llvm::ArrayRef(Parents);
  Reference VParents[] = {Reference(EmptySID, "G", InfoType::IT_record,
                                    "path::to::G::G", "path/to/G")};
  I.VirtualParents = llvm::ArrayRef(VParents);

  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record,
                        "path::to::A::r::ChildStruct", "path/to/A/r");
  I.Children.Records.push_back(ChildStruct);
  FunctionInfo F2;
  F2.Name = "OneFunction";
  I.Children.Functions.push_back(F2);

  EnumInfo E;
  E.Name = "OneEnum";
  I.Children.Enums.push_back(E);

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'r'
Path:            'path/to/A'
Namespace:
  - Type:            Namespace
    Name:            'A'
    QualName:        'A'
DefLocation:
  LineNumber:      10
  Filename:        'test.cpp'
Location:
  - LineNumber:      12
    Filename:        'test.cpp'
TagType:         Class
IsTypeDef:       true
Members:
  - Type:
      Name:            'int'
      QualName:        'int'
    Name:            'X'
    Access:          Private
    Description:
      - Kind:            FullComment
        Children:
          - Kind:            ParagraphComment
            Children:
              - Kind:            TextComment
                Text:            'Value of the thing.'
                Name:            'ParagraphComment'
Bases:
  - USR:             '0000000000000000000000000000000000000000'
    Name:            'F'
    Path:            'path/to/F'
    TagType:         Struct
    Members:
      - Type:
          Name:            'int'
          QualName:        'int'
        Name:            'N'
        Access:          Private
    ChildFunctions:
      - USR:             '0000000000000000000000000000000000000000'
        Name:            'InheritedFunctionOne'
        ReturnType:      {}
        Access:          Public
    IsVirtual:       true
    Access:          Public
    IsParent:        true
Parents:
  - Type:            Record
    Name:            'F'
VirtualParents:
  - Type:            Record
    Name:            'G'
    QualName:        'path::to::G::G'
    Path:            'path/to/G'
ChildRecords:
  - Type:            Record
    Name:            'ChildStruct'
    QualName:        'path::to::A::r::ChildStruct'
    Path:            'path/to/A/r'
ChildFunctions:
  - USR:             '0000000000000000000000000000000000000000'
    Name:            'OneFunction'
    ReturnType:      {}
    Access:          Public
ChildEnums:
  - USR:             '0000000000000000000000000000000000000000'
    Name:            'OneEnum'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(YAMLGeneratorTest, emitFunctionYAML) {
  FunctionInfo I;
  I.Name = "f";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  I.Access = AccessSpecifier::AS_none;

  I.ReturnType = TypeInfo(Reference(EmptySID, "void", InfoType::IT_default));

  FieldTypeInfo P1(TypeInfo("int"), "P");
  FieldTypeInfo D(TypeInfo("double"), "D");
  D.DefaultValue = "2.0 * M_PI";
  FieldTypeInfo Params[] = {std::move(P1), std::move(D)};
  I.Params = llvm::ArrayRef(Params);
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'f'
Namespace:
  - Type:            Namespace
    Name:            'A'
    QualName:        'A'
DefLocation:
  LineNumber:      10
  Filename:        'test.cpp'
Location:
  - LineNumber:      12
    Filename:        'test.cpp'
IsMethod:        true
Parent:
  Type:            Record
  Name:            'Parent'
  QualName:        'Parent'
Params:
  - Type:
      Name:            'int'
      QualName:        'int'
    Name:            'P'
  - Type:
      Name:            'double'
      QualName:        'double'
    Name:            'D'
    DefaultValue:    '2.0 * M_PI'
ReturnType:
  Type:
    Name:            'void'
    QualName:        'void'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

// Tests the equivalent of:
// namespace A {
// enum e { X };
// }
TEST_F(YAMLGeneratorTest, emitSimpleEnumYAML) {
  EnumInfo I;
  I.Name = "e";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  I.DefLoc = Location(10, 10, "test.cpp");
  Location Loc1(12, 12, "test.cpp");
  I.Loc.push_back(Loc1);

  EnumValueInfo EV[] = {EnumValueInfo("X")};
  I.Members = llvm::ArrayRef(EV);
  I.Scoped = false;

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'e'
Namespace:
  - Type:            Namespace
    Name:            'A'
    QualName:        'A'
DefLocation:
  LineNumber:      10
  Filename:        'test.cpp'
Location:
  - LineNumber:      12
    Filename:        'test.cpp'
Members:
  - Name:            'X'
    Value:           '0'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

// Tests the equivalent of:
// enum class e : short { X = FOO_BAR + 2 };
TEST_F(YAMLGeneratorTest, enumTypedScopedEnumYAML) {
  EnumInfo I;
  I.Name = "e";

  EnumValueInfo EV[] = {EnumValueInfo("X", "-9876", "FOO_BAR + 2")};
  I.Members = llvm::ArrayRef(EV);
  I.Scoped = true;
  I.BaseType = TypeInfo("short");

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'e'
Scoped:          true
BaseType:
  Type:
    Name:            'short'
    QualName:        'short'
Members:
  - Name:            'X'
    Value:           '-9876'
    Expr:            'FOO_BAR + 2'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(YAMLGeneratorTest, enumTypedefYAML) {
  TypedefInfo I;
  I.Name = "MyUsing";
  I.Underlying = TypeInfo("int");
  I.IsUsing = true;

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'MyUsing'
Underlying:
  Name:            'int'
  QualName:        'int'
IsUsing:         true
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(YAMLGeneratorTest, emitCommentYAML) {
  FunctionInfo I;
  I.Name = "f";
  I.DefLoc = Location(10, 10, "test.cpp");
  I.ReturnType = TypeInfo("void");
  FieldTypeInfo Params[] = {FieldTypeInfo(TypeInfo("int"), "I"),
                            FieldTypeInfo(TypeInfo("int"), "J")};
  I.Params = llvm::ArrayRef(Params);
  I.Access = AccessSpecifier::AS_none;

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

  I.Description.push_back(Top);

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'f'
Description:
  - Kind:            FullComment
    Children:
      - Kind:            ParagraphComment
        Children:
          - Kind:            TextComment
      - Kind:            ParagraphComment
        Children:
          - Kind:            TextComment
            Text:            ' Brief description.'
            Name:            'ParagraphComment'
      - Kind:            ParagraphComment
        Children:
          - Kind:            TextComment
            Text:            ' Extended description that'
          - Kind:            TextComment
            Text:            ' continues onto the next line.'
      - Kind:            ParagraphComment
        Children:
          - Kind:            TextComment
          - Kind:            HTMLStartTagComment
            Name:            'ul'
            AttrKeys:
              - 'class'
            AttrValues:
              - 'test'
          - Kind:            HTMLStartTagComment
            Name:            'li'
          - Kind:            TextComment
            Text:            ' Testing.'
          - Kind:            HTMLEndTagComment
            Name:            'ul'
            SelfClosing:     true
      - Kind:            VerbatimBlockComment
        Name:            'verbatim'
        CloseName:       'endverbatim'
        Children:
          - Kind:            VerbatimBlockLineComment
            Text:            ' The description continues.'
      - Kind:            ParamCommandComment
        Direction:       '[out]'
        ParamName:       'I'
        Explicit:        true
        Children:
          - Kind:            ParagraphComment
            Children:
              - Kind:            TextComment
              - Kind:            TextComment
                Text:            ' is a parameter.'
      - Kind:            ParamCommandComment
        Direction:       '[in]'
        ParamName:       'J'
        Children:
          - Kind:            ParagraphComment
            Children:
              - Kind:            TextComment
                Text:            ' is a parameter.'
              - Kind:            TextComment
      - Kind:            BlockCommandComment
        Name:            'return'
        Explicit:        true
        Children:
          - Kind:            ParagraphComment
            Children:
              - Kind:            TextComment
                Text:            'void'
DefLocation:
  LineNumber:      10
  Filename:        'test.cpp'
Params:
  - Type:
      Name:            'int'
      QualName:        'int'
    Name:            'I'
  - Type:
      Name:            'int'
      QualName:        'int'
    Name:            'J'
ReturnType:
  Type:
    Name:            'void'
    QualName:        'void'
...
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

} // namespace doc
} // namespace clang
