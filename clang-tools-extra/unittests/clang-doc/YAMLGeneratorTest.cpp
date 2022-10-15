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

std::unique_ptr<Generator> getYAMLGenerator() {
  auto G = doc::findGeneratorByName("yaml");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

TEST(YAMLGeneratorTest, emitNamespaceYAML) {
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Path = "path/to/A";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.Children.Namespaces.emplace_back(EmptySID, "ChildNamespace",
                                     InfoType::IT_namespace,
                                     "path/to/A/Namespace");
  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record,
                                  "path/to/A/Namespace");
  I.Children.Functions.emplace_back();
  I.Children.Functions.back().Name = "OneFunction";
  I.Children.Functions.back().Access = AccessSpecifier::AS_none;
  I.Children.Enums.emplace_back();
  I.Children.Enums.back().Name = "OneEnum";

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'Namespace'
Path:            'path/to/A'
Namespace:
  - Type:            Namespace
    Name:            'A'
ChildNamespaces:
  - Type:            Namespace
    Name:            'ChildNamespace'
    Path:            'path/to/A/Namespace'
ChildRecords:
  - Type:            Record
    Name:            'ChildStruct'
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

TEST(YAMLGeneratorTest, emitRecordYAML) {
  RecordInfo I;
  I.Name = "r";
  I.Path = "path/to/A";
  I.IsTypeDef = true;
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back(TypeInfo("int", "path/to/int"), "X",
                         AccessSpecifier::AS_private);

  // Member documentation.
  CommentInfo TopComment;
  TopComment.Kind = "FullComment";
  TopComment.Children.emplace_back(std::make_unique<CommentInfo>());
  CommentInfo *Brief = TopComment.Children.back().get();
  Brief->Kind = "ParagraphComment";
  Brief->Children.emplace_back(std::make_unique<CommentInfo>());
  Brief->Children.back()->Kind = "TextComment";
  Brief->Children.back()->Name = "ParagraphComment";
  Brief->Children.back()->Text = "Value of the thing.";
  I.Members.back().Description.push_back(std::move(TopComment));

  I.TagType = TagTypeKind::TTK_Class;
  I.Bases.emplace_back(EmptySID, "F", "path/to/F", true,
                       AccessSpecifier::AS_public, true);
  I.Bases.back().Children.Functions.emplace_back();
  I.Bases.back().Children.Functions.back().Name = "InheritedFunctionOne";
  I.Bases.back().Members.emplace_back(TypeInfo("int", "path/to/int"), "N",
                                      AccessSpecifier::AS_private);
  // F is in the global namespace
  I.Parents.emplace_back(EmptySID, "F", InfoType::IT_record, "");
  I.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record,
                                "path/to/G");

  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record,
                                  "path/to/A/r");
  I.Children.Functions.emplace_back();
  I.Children.Functions.back().Name = "OneFunction";
  I.Children.Enums.emplace_back();
  I.Children.Enums.back().Name = "OneEnum";

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'r'
Path:            'path/to/A'
Namespace:
  - Type:            Namespace
    Name:            'A'
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
      Path:            'path/to/int'
    Name:            'X'
    Access:          Private
    Description:
      - Kind:            'FullComment'
        Children:
          - Kind:            'ParagraphComment'
            Children:
              - Kind:            'TextComment'
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
          Path:            'path/to/int'
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
    Path:            'path/to/G'
ChildRecords:
  - Type:            Record
    Name:            'ChildStruct'
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

TEST(YAMLGeneratorTest, emitFunctionYAML) {
  FunctionInfo I;
  I.Name = "f";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Access = AccessSpecifier::AS_none;

  I.ReturnType = TypeInfo(
      Reference(EmptySID, "void", InfoType::IT_default, "path/to/void"));
  I.Params.emplace_back(TypeInfo("int", "path/to/int"), "P");
  I.Params.emplace_back(TypeInfo("double", "path/to/double"), "D");
  I.Params.back().DefaultValue = "2.0 * M_PI";
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'f'
Namespace:
  - Type:            Namespace
    Name:            'A'
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
Params:
  - Type:
      Name:            'int'
      Path:            'path/to/int'
    Name:            'P'
  - Type:
      Name:            'double'
      Path:            'path/to/double'
    Name:            'D'
    DefaultValue:    '2.0 * M_PI'
ReturnType:
  Type:
    Name:            'void'
    Path:            'path/to/void'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

// Tests the equivalent of:
// namespace A {
// enum e { X };
// }
TEST(YAMLGeneratorTest, emitSimpleEnumYAML) {
  EnumInfo I;
  I.Name = "e";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back("X");
  I.Scoped = false;

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'e'
Namespace:
  - Type:            Namespace
    Name:            'A'
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
TEST(YAMLGeneratorTest, enumTypedScopedEnumYAML) {
  EnumInfo I;
  I.Name = "e";

  I.Members.emplace_back("X", "-9876", "FOO_BAR + 2");
  I.Scoped = true;
  I.BaseType = TypeInfo("short");

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'e'
Scoped:          true
BaseType:
  Type:
    Name:            'short'
Members:
  - Name:            'X'
    Value:           '-9876'
    Expr:            'FOO_BAR + 2'
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST(YAMLGeneratorTest, enumTypedefYAML) {
  TypedefInfo I;
  I.Name = "MyUsing";
  I.Underlying = TypeInfo("int");
  I.IsUsing = true;

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'MyUsing'
Underlying:
  Name:            'int'
IsUsing:         true
...
)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST(YAMLGeneratorTest, emitCommentYAML) {
  FunctionInfo I;
  I.Name = "f";
  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.ReturnType = TypeInfo("void");
  I.Params.emplace_back(TypeInfo("int"), "I");
  I.Params.emplace_back(TypeInfo("int"), "J");
  I.Access = AccessSpecifier::AS_none;

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

  I.Description.emplace_back(std::move(Top));

  auto G = getYAMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected =
      R"raw(---
USR:             '0000000000000000000000000000000000000000'
Name:            'f'
Description:
  - Kind:            'FullComment'
    Children:
      - Kind:            'ParagraphComment'
        Children:
          - Kind:            'TextComment'
      - Kind:            'ParagraphComment'
        Children:
          - Kind:            'TextComment'
            Text:            ' Brief description.'
            Name:            'ParagraphComment'
      - Kind:            'ParagraphComment'
        Children:
          - Kind:            'TextComment'
            Text:            ' Extended description that'
          - Kind:            'TextComment'
            Text:            ' continues onto the next line.'
      - Kind:            'ParagraphComment'
        Children:
          - Kind:            'TextComment'
          - Kind:            'HTMLStartTagComment'
            Name:            'ul'
            AttrKeys:
              - 'class'
            AttrValues:
              - 'test'
          - Kind:            'HTMLStartTagComment'
            Name:            'li'
          - Kind:            'TextComment'
            Text:            ' Testing.'
          - Kind:            'HTMLEndTagComment'
            Name:            'ul'
            SelfClosing:     true
      - Kind:            'VerbatimBlockComment'
        Name:            'verbatim'
        CloseName:       'endverbatim'
        Children:
          - Kind:            'VerbatimBlockLineComment'
            Text:            ' The description continues.'
      - Kind:            'ParamCommandComment'
        Direction:       '[out]'
        ParamName:       'I'
        Explicit:        true
        Children:
          - Kind:            'ParagraphComment'
            Children:
              - Kind:            'TextComment'
              - Kind:            'TextComment'
                Text:            ' is a parameter.'
      - Kind:            'ParamCommandComment'
        Direction:       '[in]'
        ParamName:       'J'
        Children:
          - Kind:            'ParagraphComment'
            Children:
              - Kind:            'TextComment'
                Text:            ' is a parameter.'
              - Kind:            'TextComment'
      - Kind:            'BlockCommandComment'
        Name:            'return'
        Explicit:        true
        Children:
          - Kind:            'ParagraphComment'
            Children:
              - Kind:            'TextComment'
                Text:            'void'
DefLocation:
  LineNumber:      10
  Filename:        'test.cpp'
Params:
  - Type:
      Name:            'int'
    Name:            'I'
  - Type:
      Name:            'int'
    Name:            'J'
ReturnType:
  Type:
    Name:            'void'
...
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

} // namespace doc
} // namespace clang
