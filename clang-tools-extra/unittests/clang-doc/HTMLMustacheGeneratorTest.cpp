//===-- clang-doc/HTMLMustacheGeneratorTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocTest.h"
#include "Generators.h"
#include "Representation.h"
#include "config.h"
#include "support/Utils.h"
#include "clang/Basic/Version.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace testing;
using namespace clang;
using namespace clang::doc;

static const std::string ClangDocVersion = getClangToolFullVersion("clang-doc");

static std::unique_ptr<Generator> getHTMLMustacheGenerator() {
  auto G = findGeneratorByName("mustache");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

static ClangDocContext
getClangDocContext(std::vector<std::string> UserStylesheets = {},
                   StringRef RepositoryUrl = "",
                   StringRef RepositoryLinePrefix = "", StringRef Base = "") {
  ClangDocContext CDCtx{
      {},   "test-project", {}, {}, {}, RepositoryUrl, RepositoryLinePrefix,
      Base, UserStylesheets};
  CDCtx.UserStylesheets.insert(CDCtx.UserStylesheets.begin(), "");
  CDCtx.JsScripts.emplace_back("");
  return CDCtx;
}

static void verifyFileContents(const Twine &Path, StringRef Contents) {
  auto Buffer = MemoryBuffer::getFile(Path);
  ASSERT_TRUE((bool)Buffer);
  StringRef Data = Buffer.get()->getBuffer();
  ASSERT_EQ(Data, Contents);
}

TEST(HTMLMustacheGeneratorTest, createResources) {
  auto G = getHTMLMustacheGenerator();
  ASSERT_THAT(G, NotNull()) << "Could not find HTMLMustacheGenerator";
  ClangDocContext CDCtx = getClangDocContext();
  EXPECT_THAT_ERROR(G->createResources(CDCtx), Failed())
      << "Empty UserStylesheets or JsScripts should fail!";

  unittest::TempDir RootTestDirectory("createResourcesTest", /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  unittest::TempFile CSS("clang-doc-mustache", "css", "CSS");
  unittest::TempFile JS("mustache", "js", "JavaScript");

  CDCtx.UserStylesheets[0] = CSS.path();
  CDCtx.JsScripts[0] = JS.path();

  EXPECT_THAT_ERROR(G->createResources(CDCtx), Succeeded())
      << "Failed to create resources with valid UserStylesheets and JsScripts";
  {
    SmallString<256> PathBuf;
    llvm::sys::path::append(PathBuf, RootTestDirectory.path(),
                            "clang-doc-mustache.css");
    verifyFileContents(PathBuf, "CSS");
  }

  {
    SmallString<256> PathBuf;
    llvm::sys::path::append(PathBuf, RootTestDirectory.path(), "mustache.js");
    verifyFileContents(PathBuf, "JavaScript");
  }
}

TEST(HTMLMustacheGeneratorTest, generateDocs) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();

  unittest::TempDir RootTestDirectory("generateDocsTest", /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";
}

TEST(HTMLMustacheGeneratorTest, generateDocsForInfo) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.Children.Namespaces.emplace_back(EmptySID, "ChildNamespace",
                                     InfoType::IT_namespace,
                                     "Namespace::ChildNamespace", "Namespace");
  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record,
                                  "Namespace::ChildStruct", "Namespace");
  I.Children.Functions.emplace_back();
  I.Children.Functions.back().Access = AccessSpecifier::AS_none;
  I.Children.Functions.back().Name = "OneFunction";
  I.Children.Enums.emplace_back();

  unittest::TempDir RootTestDirectory("generateDocForInfoTest",
                                      /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  // FIXME: This is a terrible hack, since we can't initialize the templates
  // directly. We'll need to update the interfaces so that we can call
  // SetupTemplateFiles() from outsize of HTMLMustacheGenerator.cpp
  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";

  EXPECT_THAT_ERROR(G->generateDocForInfo(&I, Actual, CDCtx), Succeeded());

  std::string Expected = R"raw(<!DOCTYPE html>
<html lang="en-US">
    <head>
        <meta charset="utf-8"/>
        <title>namespace Namespace</title>
        <link rel="stylesheet" type="text/css" href="../clang-doc-mustache.css"/>
        <link rel="stylesheet" type="text/css" href="../"/>
        <script src="../mustache-index.js"></script>
        <script src="../"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
    </head>
    <body>
        <nav class="navbar">
            Navbar
        </nav>
        <main>
            <div class="container">
                <div class="sidebar">
                        Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
                        sed do eiusmod tempor incididunt ut labore et dolore magna 
                        aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco 
                        laboris nisi ut aliquip ex ea commodo consequat. 
                        Duis aute irure dolor in reprehenderit in voluptate velit esse 
                        cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat 
                        cupidatat non proident, sunt in culpa qui officia deserunt mollit 
                        anim id est laborum
                </div>
                <div class="resizer" id="resizer"></div>
                <div class="content">
                    Content
                </div>
            </div>
        </main>
    </body>
</html>
)raw";
  EXPECT_EQ(Actual.str(), Expected);
}

TEST(HTMLMustacheGeneratorTest, emitRecordHTML) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);

  unittest::TempDir RootTestDirectory("emitRecordHTML",
                                      /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  // FIXME: This is a terrible hack, since we can't initialize the templates
  // directly. We'll need to update the interfaces so that we can call
  // SetupTemplateFiles() from outsize of HTMLMustacheGenerator.cpp
  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";

  CDCtx.RepositoryUrl = "http://www.repository.com";

  RecordInfo I;
  I.Name = "r";
  I.Path = "X/Y/Z";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, 10, "dir/test.cpp", true);
  I.Loc.emplace_back(12, 12, "test.cpp");

  SmallString<16> PathTo;
  llvm::sys::path::native("path/to", PathTo);
  I.Members.emplace_back(clang::doc::TypeInfo("int"), "X",
                         AccessSpecifier::AS_private);
  I.TagType = TagTypeKind::Class;
  I.Parents.emplace_back(EmptySID, "F", InfoType::IT_record, "F", PathTo);
  I.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);

  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record,
                                  "X::Y::Z::r::ChildStruct", "X/Y/Z/r");
  I.Children.Functions.emplace_back();
  I.Children.Functions.back().Name = "OneFunction";
  I.Children.Enums.emplace_back();
  I.Children.Enums.back().Name = "OneEnum";

  EXPECT_THAT_ERROR(G->generateDocForInfo(&I, Actual, CDCtx), Succeeded());

  std::string Expected = R"raw(<!DOCTYPE html>
<html lang="en-US">
<head>
    <meta charset="utf-8"/>
    <title>r</title>
        <link rel="stylesheet" type="text/css" href="../../../clang-doc-mustache.css"/>
        <link rel="stylesheet" type="text/css" href="../../../"/>
        <script src="../../../mustache-index.js"></script>
        <script src="../../../"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
</head>
<body>
<nav class="navbar">
    <div class="navbar__container">
            <div class="navbar__logo">
                test-project
            </div>
        <div class="navbar__menu">
            <ul class="navbar__links">
                <li class="navbar__item">
                    <a href="/" class="navbar__link">Namespace</a>
                </li>
                <li class="navbar__item">
                    <a href="/" class="navbar__link">Class</a>
                </li>
            </ul>
        </div>
    </div>
</nav>
<main>
    <div class="container">
        <div class="sidebar">
            <h2>class r</h2>
            <ul>
                    <li class="sidebar-section">
                        <a class="sidebar-item" href="#PublicMethods">Protected Members</a>
                    </li>
                    <ul>
                            <li class="sidebar-item-container">
                                <a class="sidebar-item" href="#X">X</a>
                            </li>
                    </ul>
                <li class="sidebar-section">
                    <a class="sidebar-item" href="#PublicMethods">Public Method</a>
                </li>
                <ul>
                    <li class="sidebar-item-container">
                        <a class="sidebar-item" href="#0000000000000000000000000000000000000000">OneFunction</a>
                    </li>
                </ul>
                <li class="sidebar-section">
                    <a class="sidebar-item" href="#Enums">Enums</a>
                </li>
                <ul>
                    <li class="sidebar-item-container">
                        <a class="sidebar-item" href="#0000000000000000000000000000000000000000">enum OneEnum</a>
                    </li>
                </ul>
                <li class="sidebar-section">
                    <a class="sidebar-item" href="#Classes">Inner Classes</a>
                </li>
                <ul>
                    <li class="sidebar-item-container">
                        <a class="sidebar-item" href="#0000000000000000000000000000000000000000">ChildStruct</a>
                    </li>
                </ul>
            </ul>
        </div>
        <div class="resizer" id="resizer"></div>
        <div class="content">
            <section class="hero section-container">
                <div class="hero__title">
                    <h1 class="hero__title-large">class r</h1>
                </div>
            </section>
            <section id="ProtectedMembers" class="section-container">
                <h2>Protected Members</h2>
                <div>
                    <div id="X" class="delimiter-container">
                        <pre>
<code class="language-cpp code-clang-doc" >int X</code>
                        </pre>
                    </div>
                </div>
            </section>
            <section id="PublicMethods" class="section-container">
                <h2>Public Methods</h2>
                <div>
<div class="delimiter-container">
    <div id="0000000000000000000000000000000000000000">
        <pre>
            <code class="language-cpp code-clang-doc">
 OneFunction ()
            </code>
        </pre>
    </div>
</div>
                </div>
            </section>
            <section id="Enums" class="section-container">
                <h2>Enumerations</h2>
                <div>
<div id="0000000000000000000000000000000000000000" class="delimiter-container">
    <div>
        <pre>
            <code class="language-cpp code-clang-doc">
enum OneEnum
            </code>
        </pre>
    </div>
    <table class="table-wrapper">
        <tbody>
        <tr>
            <th>Name</th>
            <th>Value</th>
        </tr>
        </tbody>
    </table>
</div>
                </div>
            </section>
            <section id="Classes" class="section-container">
                <h2>Inner Classes</h2>
                <ul class="class-container">
                    <li id="0000000000000000000000000000000000000000" style="max-height: 40px;">
<a href="../../../X/Y/Z/r/ChildStruct.html"><pre><code class="language-cpp code-clang-doc" >class ChildStruct</code></pre></a>
                    </li>
                </ul>
            </section>
        </div>
    </div>
</main>
</body>
</html>
)raw";
  EXPECT_EQ(Actual.str(), Expected);
}

TEST(HTMLGeneratorTest, emitFunctionHTML) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);

  unittest::TempDir RootTestDirectory("emitRecordHTML",
                                      /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  // FIXME: This is a terrible hack, since we can't initialize the templates
  // directly. We'll need to update the interfaces so that we can call
  // SetupTemplateFiles() from outsize of HTMLMustacheGenerator.cpp
  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";

  CDCtx.RepositoryUrl = "http://www.repository.com";

  FunctionInfo I;
  I.Name = "f";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, 10, "dir/test.cpp", true);
  I.Loc.emplace_back(12, 12, "test.cpp");

  I.Access = AccessSpecifier::AS_none;

  SmallString<16> PathTo;
  llvm::sys::path::native("path/to", PathTo);
  I.ReturnType = doc::TypeInfo(
      Reference(EmptySID, "float", InfoType::IT_default, "float", PathTo));
  I.Params.emplace_back(doc::TypeInfo("int", PathTo), "P");
  I.IsMethod = true;
  I.Parent = Reference(EmptySID, "Parent", InfoType::IT_record);

  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  std::string Expected = R"raw(IT_Function
)raw";

  // FIXME: Functions are not handled yet.
  EXPECT_EQ(Expected, Actual.str());
}

TEST(HTMLMustacheGeneratorTest, emitEnumHTML) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);

  unittest::TempDir RootTestDirectory("emitEnumHTML",
                                      /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  // FIXME: This is a terrible hack, since we can't initialize the templates
  // directly. We'll need to update the interfaces so that we can call
  // SetupTemplateFiles() from outsize of HTMLMustacheGenerator.cpp
  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";

  CDCtx.RepositoryUrl = "http://www.repository.com";

  EnumInfo I;
  I.Name = "e";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.DefLoc = Location(10, 10, "test.cpp", true);
  I.Loc.emplace_back(12, 12, "test.cpp");

  I.Members.emplace_back("X");
  I.Scoped = true;

  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);

  std::string Expected = R"raw(IT_enum
)raw";

  // FIXME: Enums are not handled yet.
  EXPECT_EQ(Expected, Actual.str());
}

TEST(HTMLMustacheGeneratorTest, emitCommentHTML) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);

  unittest::TempDir RootTestDirectory("emitCommentHTML",
                                      /*Unique=*/true);
  CDCtx.OutDirectory = RootTestDirectory.path();

  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  // FIXME: This is a terrible hack, since we can't initialize the templates
  // directly. We'll need to update the interfaces so that we can call
  // SetupTemplateFiles() from outsize of HTMLMustacheGenerator.cpp
  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";

  CDCtx.RepositoryUrl = "http://www.repository.com";

  FunctionInfo I;
  I.Name = "f";
  I.DefLoc = Location(10, 10, "test.cpp", true);
  I.ReturnType = doc::TypeInfo("void");
  I.Params.emplace_back(doc::TypeInfo("int"), "I");
  I.Params.emplace_back(doc::TypeInfo("int"), "J");
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
  CommentInfo *Entities = Top.Children.back().get();
  Entities->Kind = "ParagraphComment";
  Entities->Children.emplace_back(std::make_unique<CommentInfo>());
  Entities->Children.back()->Kind = "TextComment";
  Entities->Children.back()->Name = "ParagraphComment";
  Entities->Children.back()->Text =
      " Comment with html entities: &, <, >, \", \'.";

  I.Description.emplace_back(std::move(Top));

  auto Err = G->generateDocForInfo(&I, Actual, CDCtx);
  assert(!Err);
  std::string Expected = R"raw(IT_Function
)raw";

  // FIXME: Functions are not handled yet.
  EXPECT_EQ(Expected, Actual.str());
}
