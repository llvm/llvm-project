//===- SynthesisTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests synthesis API for syntax trees.
//
//===----------------------------------------------------------------------===//

#include "TreeTestBase.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::syntax;

namespace {

class SynthesisTest : public SyntaxTreeTest {
protected:
  ::testing::AssertionResult treeDumpEqual(syntax::Node *Root, StringRef Dump) {
    if (!Root)
      return ::testing::AssertionFailure()
             << "Root was not built successfully.";

    auto Actual = StringRef(Root->dump(Arena->getSourceManager())).trim().str();
    auto Expected = Dump.trim().str();
    // EXPECT_EQ shows the diff between the two strings if they are different.
    EXPECT_EQ(Expected, Actual);
    if (Actual != Expected) {
      return ::testing::AssertionFailure();
    }
    return ::testing::AssertionSuccess();
  }
};

INSTANTIATE_TEST_CASE_P(SynthesisTests, SynthesisTest,
                        ::testing::ValuesIn(allTestClangConfigs()), );

TEST_P(SynthesisTest, Leaf_Punctuation) {
  buildTree("", GetParam());

  auto *Leaf = createLeaf(*Arena, tok::comma);

  EXPECT_TRUE(treeDumpEqual(Leaf, R"txt(
',' Detached synthesized
  )txt"));
}

TEST_P(SynthesisTest, Leaf_Keyword) {
  buildTree("", GetParam());

  auto *Leaf = createLeaf(*Arena, tok::kw_if);

  EXPECT_TRUE(treeDumpEqual(Leaf, R"txt(
'if' Detached synthesized
  )txt"));
}

TEST_P(SynthesisTest, Leaf_Identifier) {
  buildTree("", GetParam());

  auto *Leaf = createLeaf(*Arena, tok::identifier, "a");

  EXPECT_TRUE(treeDumpEqual(Leaf, R"txt(
'a' Detached synthesized
  )txt"));
}

TEST_P(SynthesisTest, Leaf_Number) {
  buildTree("", GetParam());

  auto *Leaf = createLeaf(*Arena, tok::numeric_constant, "1");

  EXPECT_TRUE(treeDumpEqual(Leaf, R"txt(
'1' Detached synthesized
  )txt"));
}

TEST_P(SynthesisTest, Statement_EmptyStatement) {
  buildTree("", GetParam());

  auto *S = createEmptyStatement(*Arena);
  EXPECT_TRUE(treeDumpEqual(S, R"txt(
EmptyStatement Detached synthesized
`-';' synthesized
  )txt"));
}
} // namespace
