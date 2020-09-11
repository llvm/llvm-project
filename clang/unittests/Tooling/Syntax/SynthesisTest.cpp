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

using namespace clang;
using namespace clang::syntax;

namespace {

INSTANTIATE_TEST_CASE_P(SyntaxTreeTests, SyntaxTreeTest,
                        ::testing::ValuesIn(allTestClangConfigs()), );

TEST_P(SyntaxTreeTest, Leaf_Punctuation) {
  buildTree("", GetParam());

  auto *C = syntax::createPunctuation(*Arena, tok::comma);
  ASSERT_NE(C, nullptr);
  EXPECT_EQ(C->getToken()->kind(), tok::comma);
  EXPECT_TRUE(C->canModify());
  EXPECT_FALSE(C->isOriginal());
  EXPECT_TRUE(C->isDetached());
}

TEST_P(SyntaxTreeTest, Statement_Empty) {
  buildTree("", GetParam());

  auto *S = syntax::createEmptyStatement(*Arena);
  ASSERT_NE(S, nullptr);
  EXPECT_TRUE(S->canModify());
  EXPECT_FALSE(S->isOriginal());
  EXPECT_TRUE(S->isDetached());
}
} // namespace
