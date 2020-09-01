//===- MutationsTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests mutation API for syntax trees.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Mutations.h"
#include "TreeTestBase.h"
#include "clang/Tooling/Syntax/BuildTree.h"

using namespace clang;
using namespace clang::syntax;

namespace {

TEST_P(SyntaxTreeTest, Mutations) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  using Transformation = std::function<void(
      const llvm::Annotations & /*Input*/, syntax::TranslationUnit * /*Root*/)>;
  auto CheckTransformation = [this](std::string Input, std::string Expected,
                                    Transformation Transform) -> void {
    llvm::Annotations Source(Input);
    auto *Root = buildTree(Source.code(), GetParam());

    Transform(Source, Root);

    auto Replacements = syntax::computeReplacements(*Arena, *Root);
    auto Output = tooling::applyAllReplacements(Source.code(), Replacements);
    if (!Output) {
      ADD_FAILURE() << "could not apply replacements: "
                    << llvm::toString(Output.takeError());
      return;
    }

    EXPECT_EQ(Expected, *Output) << "input is:\n" << Input;
  };

  // Removes the selected statement. Input should have exactly one selected
  // range and it should correspond to a single statement.
  auto RemoveStatement = [this](const llvm::Annotations &Input,
                                syntax::TranslationUnit *TU) {
    auto *S = cast<syntax::Statement>(nodeByRange(Input.range(), TU));
    ASSERT_TRUE(S->canModify()) << "cannot remove a statement";
    syntax::removeStatement(*Arena, S);
    EXPECT_TRUE(S->isDetached());
    EXPECT_FALSE(S->isOriginal())
        << "node removed from tree cannot be marked as original";
  };

  std::vector<std::pair<std::string /*Input*/, std::string /*Expected*/>>
      Cases = {
          {"void test() { [[100+100;]] test(); }", "void test() {  test(); }"},
          {"void test() { if (true) [[{}]] else {} }",
           "void test() { if (true) ; else {} }"},
          {"void test() { [[;]] }", "void test() {  }"}};
  for (const auto &C : Cases)
    CheckTransformation(C.first, C.second, RemoveStatement);
}

TEST_P(SyntaxTreeTest, SynthesizedNodes) {
  buildTree("", GetParam());

  auto *C = syntax::createPunctuation(*Arena, tok::comma);
  ASSERT_NE(C, nullptr);
  EXPECT_EQ(C->token()->kind(), tok::comma);
  EXPECT_TRUE(C->canModify());
  EXPECT_FALSE(C->isOriginal());
  EXPECT_TRUE(C->isDetached());

  auto *S = syntax::createEmptyStatement(*Arena);
  ASSERT_NE(S, nullptr);
  EXPECT_TRUE(S->canModify());
  EXPECT_FALSE(S->isOriginal());
  EXPECT_TRUE(S->isDetached());
}
} // namespace
