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

// FIXME: Don't enable unit tests that can read files. Remove once we can use
// lit to test these properties.
#define ENABLE_LOCAL_TEST 0

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

#if ENABLE_LOCAL_TEST
  // FIXME: We can't read files during unit tests. Migrate to lit once
  // tool support lands.
  getMustacheHtmlFiles(CLANG_DOC_TEST_ASSET_DIR, CDCtx);

  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Succeeded())
      << "Failed to generate docs.";
#else
  EXPECT_THAT_ERROR(G->generateDocs(RootTestDirectory.path(), {}, CDCtx),
                    Failed())
      << "Failed to generate docs.";
#endif
}

TEST(HTMLGeneratorTest, emitFunctionHTML) {
#if ENABLE_LOCAL_TEST
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
#endif
}

TEST(HTMLMustacheGeneratorTest, emitEnumHTML) {
#if ENABLE_LOCAL_TEST
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
#endif
}

TEST(HTMLMustacheGeneratorTest, emitCommentHTML) {
#if ENABLE_LOCAL_TEST
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
#endif
}
