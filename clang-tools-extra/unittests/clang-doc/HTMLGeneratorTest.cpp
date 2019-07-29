//===-- clang-doc/HTMLGeneratorTest.cpp -----------------------------------===//
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

std::unique_ptr<Generator> getHTMLGenerator() {
  auto G = doc::findGeneratorByName("html");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

ClangDocContext
getClangDocContext(std::vector<std::string> UserStylesheets = {}) {
  ClangDocContext CDCtx;
  CDCtx.UserStylesheets = {UserStylesheets.begin(), UserStylesheets.end()};
  CDCtx.UserStylesheets.insert(
      CDCtx.UserStylesheets.begin(),
      "../share/clang/clang-doc-default-stylesheet.css");
  return CDCtx;
}

TEST(HTMLGeneratorTest, emitNamespaceHTML) {
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.ChildNamespaces.emplace_back(EmptySID, "ChildNamespace",
                                 InfoType::IT_namespace);
  I.ChildRecords.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record);
  I.ChildFunctions.emplace_back();
  I.ChildFunctions.back().Name = "OneFunction";
  I.ChildEnums.emplace_back();
  I.ChildEnums.back().Name = "OneEnum";

  auto G = getHTMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  ClangDocContext CDCtx = getClangDocContext({"user-provided-stylesheet.css"});
  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  std::string Expected = R"raw(<!DOCTYPE html>
<meta charset="utf-8"/>
<title>namespace Namespace</title>
<link rel="stylesheet" href="clang-doc-default-stylesheet.css"/>
<link rel="stylesheet" href="user-provided-stylesheet.css"/>
<div>
  <h1>namespace Namespace</h1>
  <h2>Namespaces</h2>
  <ul>
    <li>ChildNamespace</li>
  </ul>
  <h2>Records</h2>
  <ul>
    <li>ChildStruct</li>
  </ul>
  <h2>Functions</h2>
  <div>
    <h3>OneFunction</h3>
    <p>OneFunction()</p>
  </div>
  <h2>Enums</h2>
  <div>
    <h3>enum OneEnum</h3>
  </div>
</div>
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST(HTMLGeneratorTest, emitRecordHTML) {
  RecordInfo I;
  I.Name = "r";
  I.Path = "X/Y/Z";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  SmallString<16> PathTo;
  llvm::sys::path::native("path/to", PathTo);
  I.Members.emplace_back("int", "X/Y", "X", AccessSpecifier::AS_private);
  I.TagType = TagTypeKind::TTK_Class;
  I.Parents.emplace_back(EmptySID, "F", InfoType::IT_record, PathTo);
  I.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);

  I.ChildRecords.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record);
  I.ChildFunctions.emplace_back();
  I.ChildFunctions.back().Name = "OneFunction";
  I.ChildEnums.emplace_back();
  I.ChildEnums.back().Name = "OneEnum";

  auto G = getHTMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  ClangDocContext CDCtx = getClangDocContext();
  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  std::string Expected = R"raw(<!DOCTYPE html>
<meta charset="utf-8"/>
<title>class r</title>
<link rel="stylesheet" href="../../../clang-doc-default-stylesheet.css"/>
<div>
  <h1>class r</h1>
  <p>Defined at line 10 of test.cpp</p>
  <p>
    Inherits from 
    <a href="../../../path/to/F.html">F</a>
    , G
  </p>
  <h2>Members</h2>
  <ul>
    <li>
      private 
      <a href="../int.html">int</a>
       X
    </li>
  </ul>
  <h2>Records</h2>
  <ul>
    <li>ChildStruct</li>
  </ul>
  <h2>Functions</h2>
  <div>
    <h3>OneFunction</h3>
    <p>OneFunction()</p>
  </div>
  <h2>Enums</h2>
  <div>
    <h3>enum OneEnum</h3>
  </div>
</div>
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST(HTMLGeneratorTest, emitFunctionHTML) {
  FunctionInfo I;
  I.Name = "f";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  SmallString<16> PathTo;
  llvm::sys::path::native("path/to", PathTo);
  I.ReturnType = TypeInfo(EmptySID, "float", InfoType::IT_default, PathTo);
  I.Params.emplace_back("int", PathTo, "P");
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  auto G = getHTMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  ClangDocContext CDCtx = getClangDocContext();
  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  SmallString<16> PathToFloat;
  llvm::sys::path::native("path/to/float.html", PathToFloat);
  SmallString<16> PathToInt;
  llvm::sys::path::native("path/to/int.html", PathToInt);
  std::string Expected = R"raw(<!DOCTYPE html>
<meta charset="utf-8"/>
<title></title>
<link rel="stylesheet" href="clang-doc-default-stylesheet.css"/>
<div>
  <h3>f</h3>
  <p>
    <a href=")raw" + std::string(PathToFloat.str()) +
                         R"raw(">float</a>
     f(
    <a href=")raw" + std::string(PathToInt.str()) +
                         R"raw(">int</a>
     P)
  </p>
  <p>Defined at line 10 of test.cpp</p>
</div>
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST(HTMLGeneratorTest, emitEnumHTML) {
  EnumInfo I;
  I.Name = "e";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  I.Members.emplace_back("X");
  I.Scoped = true;

  auto G = getHTMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  ClangDocContext CDCtx = getClangDocContext();
  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  std::string Expected = R"raw(<!DOCTYPE html>
<meta charset="utf-8"/>
<title></title>
<link rel="stylesheet" href="clang-doc-default-stylesheet.css"/>
<div>
  <h3>enum class e</h3>
  <ul>
    <li>X</li>
  </ul>
  <p>Defined at line 10 of test.cpp</p>
</div>
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

TEST(HTMLGeneratorTest, emitCommentHTML) {
  FunctionInfo I;
  I.Name = "f";
  I.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  I.ReturnType = TypeInfo(EmptySID, "void", InfoType::IT_default);
  I.Params.emplace_back("int", "I");
  I.Params.emplace_back("int", "J");

  CommentInfo Top;
  Top.Kind = "FullComment";

  Top.Children.emplace_back(llvm::make_unique<CommentInfo>());
  CommentInfo *BlankLine = Top.Children.back().get();
  BlankLine->Kind = "ParagraphComment";
  BlankLine->Children.emplace_back(llvm::make_unique<CommentInfo>());
  BlankLine->Children.back()->Kind = "TextComment";

  Top.Children.emplace_back(llvm::make_unique<CommentInfo>());
  CommentInfo *Brief = Top.Children.back().get();
  Brief->Kind = "ParagraphComment";
  Brief->Children.emplace_back(llvm::make_unique<CommentInfo>());
  Brief->Children.back()->Kind = "TextComment";
  Brief->Children.back()->Name = "ParagraphComment";
  Brief->Children.back()->Text = " Brief description.";

  Top.Children.emplace_back(llvm::make_unique<CommentInfo>());
  CommentInfo *Extended = Top.Children.back().get();
  Extended->Kind = "ParagraphComment";
  Extended->Children.emplace_back(llvm::make_unique<CommentInfo>());
  Extended->Children.back()->Kind = "TextComment";
  Extended->Children.back()->Text = " Extended description that";
  Extended->Children.emplace_back(llvm::make_unique<CommentInfo>());
  Extended->Children.back()->Kind = "TextComment";
  Extended->Children.back()->Text = " continues onto the next line.";

  Top.Children.emplace_back(llvm::make_unique<CommentInfo>());
  CommentInfo *Entities = Top.Children.back().get();
  Entities->Kind = "ParagraphComment";
  Entities->Children.emplace_back(llvm::make_unique<CommentInfo>());
  Entities->Children.back()->Kind = "TextComment";
  Entities->Children.back()->Name = "ParagraphComment";
  Entities->Children.back()->Text =
      " Comment with html entities: &, <, >, \", \'.";

  I.Description.emplace_back(std::move(Top));

  auto G = getHTMLGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  ClangDocContext CDCtx = getClangDocContext();
  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  std::string Expected = R"raw(<!DOCTYPE html>
<meta charset="utf-8"/>
<title></title>
<link rel="stylesheet" href="clang-doc-default-stylesheet.css"/>
<div>
  <h3>f</h3>
  <p>void f(int I, int J)</p>
  <p>Defined at line 10 of test.cpp</p>
  <div>
    <div>
      <p> Brief description.</p>
      <p> Extended description that continues onto the next line.</p>
      <p> Comment with html entities: &amp;, &lt;, &gt;, &quot;, &apos;.</p>
    </div>
  </div>
</div>
)raw";

  EXPECT_EQ(Expected, Actual.str());
}

} // namespace doc
} // namespace clang
