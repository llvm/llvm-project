//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for a freetrie.
///
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freetrie.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::BlockRef;
using LIBC_NAMESPACE::FreeTrie;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::optional;

TEST(LlvmLibcFreeTrie, FindBestFitRoot) {
  FreeTrie trie({0, 4096});
  EXPECT_EQ(trie.find_best_fit(123), static_cast<FreeTrie::Node *>(nullptr));

  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block = *maybeBlock;
  trie.push(block);

  FreeTrie::Node *root = trie.find_best_fit(0);
  ASSERT_EQ(root->block().addr(), block.addr());
  EXPECT_EQ(trie.find_best_fit(block.inner_size() - 1), root);
  EXPECT_EQ(trie.find_best_fit(block.inner_size()), root);
  EXPECT_EQ(trie.find_best_fit(block.inner_size() + 1),
            static_cast<FreeTrie::Node *>(nullptr));
  EXPECT_EQ(trie.find_best_fit(4095), static_cast<FreeTrie::Node *>(nullptr));
}

TEST(LlvmLibcFreeTrie, FindBestFitLower) {
  byte mem[4096];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef lower = *maybeBlock;
  maybeBlock = lower.split(512);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef root = *maybeBlock;

  FreeTrie trie({0, 4096});
  trie.push(root);
  trie.push(lower);

  EXPECT_EQ(trie.find_best_fit(0)->block().addr(), lower.addr());
}

TEST(LlvmLibcFreeTrie, FindBestFitUpper) {
  byte mem[4096];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef root = *maybeBlock;
  maybeBlock = root.split(512);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef upper = *maybeBlock;

  FreeTrie trie({0, 4096});
  trie.push(root);
  trie.push(upper);

  EXPECT_EQ(trie.find_best_fit(root.inner_size() + 1)->block().addr(),
            upper.addr());
  // The upper subtrie should be skipped if it could not contain a better fit.
  EXPECT_EQ(trie.find_best_fit(root.inner_size() - 1)->block().addr(),
            root.addr());
}

TEST(LlvmLibcFreeTrie, FindBestFitLowerAndUpper) {
  byte mem[4096];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef root = *maybeBlock;
  maybeBlock = root.split(1024);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef lower = *maybeBlock;
  maybeBlock = lower.split(128);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef upper = *maybeBlock;

  FreeTrie trie({0, 4096});
  trie.push(root);
  trie.push(lower);
  trie.push(upper);

  // The lower subtrie is examined first.
  EXPECT_EQ(trie.find_best_fit(0)->block().addr(), lower.addr());
  // The upper subtrie is examined if there are no fits found in the upper
  // subtrie.
  EXPECT_EQ(trie.find_best_fit(2048)->block().addr(), upper.addr());
}

TEST(LlvmLibcFreeTrie, Remove) {
  byte mem[4096];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef small1 = *maybeBlock;
  maybeBlock = small1.split(512);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef small2 = *maybeBlock;
  maybeBlock = small2.split(512);
  ASSERT_TRUE(maybeBlock.has_value());
  ASSERT_EQ(small1.inner_size(), small2.inner_size());
  BlockRef large = *maybeBlock;

  // Removing the root empties the trie.
  FreeTrie trie({0, 4096});
  trie.push(large);
  FreeTrie::Node *large_node = trie.find_best_fit(0);
  ASSERT_EQ(large_node->block().addr(), large.addr());
  trie.remove(large_node);
  ASSERT_TRUE(trie.empty());

  // Removing the head of a trie list preserves the trie structure.
  trie.push(small1);
  trie.push(small2);
  trie.push(large);
  trie.remove(trie.find_best_fit(small1.inner_size()));
  EXPECT_EQ(trie.find_best_fit(large.inner_size())->block().addr(),
            large.addr());
  trie.remove(trie.find_best_fit(small1.inner_size()));
  EXPECT_EQ(trie.find_best_fit(large.inner_size())->block().addr(),
            large.addr());
}
