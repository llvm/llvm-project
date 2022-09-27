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

std::unique_ptr<Generator> getMDGenerator() {
  auto G = doc::findGeneratorByName("md");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

TEST(MDGeneratorTest, emitNamespaceMD) {
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.ChildNamespaces.emplace_back(EmptySID, "ChildNamespace",
                                 InfoType::IT_namespace);
  I.ChildRecords.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record);
  I.ChildFunctions.emplace_back();
  I.ChildFunctions.back().Name = "OneFunction";
  I.ChildFunctions.back().Access = AccessSpecifier::AS_none;
  I.ChildEnums.emplace_back();
  I.ChildEnums.back().Name = "OneEnum";

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
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

--





)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST(MDGeneratorTest, emitRecordMD) {
  RecordInfo I;
  I.Name = "r";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back(TypeInfo("int"), "X", AccessSpecifier::AS_private);
  I.TagType = TagTypeKind::TTK_Class;
  I.Parents.emplace_back(EmptySID, "F", InfoType::IT_record);
  I.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);

  I.ChildRecords.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record);
  I.ChildFunctions.emplace_back();
  I.ChildFunctions.back().Name = "OneFunction";
  I.ChildEnums.emplace_back();
  I.ChildEnums.back().Name = "OneEnum";

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
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

--





)raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST(MDGeneratorTest, emitFunctionMD) {
  FunctionInfo I;
  I.Name = "f";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Access = AccessSpecifier::AS_none;

  I.ReturnType = TypeInfo("void");
  I.Params.emplace_back(TypeInfo("int"), "P");
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected = R"raw(### f

*void f(int P)*

*Defined at test.cpp#10*

)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST(MDGeneratorTest, emitEnumMD) {
  EnumInfo I;
  I.Name = "e";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back("X");
  I.Scoped = true;

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected = R"raw(| enum class e |

--

| X |


*Defined at test.cpp#10*

)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST(MDGeneratorTest, emitCommentMD) {
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

  auto G = getMDGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
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

**I** [out]

**J**

**return**void

)raw";

  EXPECT_EQ(Expected, Actual.str());
}

} // namespace doc
} // namespace clang
