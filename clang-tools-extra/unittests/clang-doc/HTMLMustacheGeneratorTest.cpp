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
#include "clang/Basic/Version.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace testing;
using namespace clang::doc;

static const std::string ClangDocVersion =
    clang::getClangToolFullVersion("clang-doc");

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

TEST(HTMLMustacheGeneratorTest, createResources) {
  auto G = getHTMLMustacheGenerator();
  ASSERT_THAT(G, NotNull()) << "Could not find HTMLMustacheGenerator";
  ClangDocContext CDCtx = getClangDocContext();

  EXPECT_THAT_ERROR(G->createResources(CDCtx), Succeeded())
      << "Failed to create resources.";
}

TEST(HTMLMustacheGeneratorTest, generateDocs) {
  auto G = getHTMLMustacheGenerator();
  assert(G && "Could not find HTMLMustacheGenerator");
  ClangDocContext CDCtx = getClangDocContext();

  StringRef RootDir = "";
  EXPECT_THAT_ERROR(G->generateDocs(RootDir, {}, CDCtx), Succeeded())
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
  I.Children.Functions.back().Access = clang::AccessSpecifier::AS_none;
  I.Children.Functions.back().Name = "OneFunction";
  I.Children.Enums.emplace_back();

  EXPECT_THAT_ERROR(G->generateDocForInfo(&I, Actual, CDCtx), Succeeded())
      << "Failed to generate docs.";

  std::string Expected = R"raw()raw";
  EXPECT_THAT(Actual.str(), Eq(Expected));
}
