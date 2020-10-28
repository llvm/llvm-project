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

class MutationTest : public SyntaxTreeTest {
protected:
  using Transformation = std::function<void(const llvm::Annotations & /*Input*/,
                                            TranslationUnit * /*Root*/)>;
  void CheckTransformation(Transformation Transform, std::string Input,
                           std::string Expected) {
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
  Transformation RemoveStatement = [this](const llvm::Annotations &Input,
                                          TranslationUnit *Root) {
    auto *S = cast<syntax::Statement>(nodeByRange(Input.range(), Root));
    ASSERT_TRUE(S->canModify()) << "cannot remove a statement";
    syntax::removeStatement(*Arena, S);
    EXPECT_TRUE(S->isDetached());
    EXPECT_FALSE(S->isOriginal())
        << "node removed from tree cannot be marked as original";
  };
};

INSTANTIATE_TEST_CASE_P(SyntaxTreeTests, MutationTest,
                        ::testing::ValuesIn(allTestClangConfigs()), );

TEST_P(MutationTest, RemoveStatement_InCompound) {
  CheckTransformation(RemoveStatement, "void test() { [[100+100;]] test(); }",
                      "void test() {  test(); }");
}

TEST_P(MutationTest, RemoveStatement_InCompound_Empty) {
  CheckTransformation(RemoveStatement, "void test() { [[;]] }",
                      "void test() {  }");
}

TEST_P(MutationTest, RemoveStatement_LeaveEmpty) {
  CheckTransformation(RemoveStatement, "void test() { if (1) [[{}]] else {} }",
                      "void test() { if (1) ; else {} }");
}
} // namespace
