//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for a freelist.
///
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freelist.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::BlockRef;
using LIBC_NAMESPACE::FreeList;
using LIBC_NAMESPACE::FreeListSecrets;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::optional;

constexpr FreeListSecrets get_test_secrets() {
#if LIBC_COPT_HARDEN_FREELIST
  return FreeListSecrets{0x1234, 0x5678, 0x9abc};
#else
  return FreeListSecrets{};
#endif
}

TEST(LlvmLibcFreeList, FreeList) {
  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block1 = *maybeBlock;

  maybeBlock = block1.split(128);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block2 = *maybeBlock;

  maybeBlock = block2.split(128);
  ASSERT_TRUE(maybeBlock.has_value());

  FreeList list;
  list.push(block1, get_test_secrets());
  ASSERT_FALSE(list.empty());
  EXPECT_EQ(list.front().addr(), block1.addr());

  list.push(block2, get_test_secrets());
  EXPECT_EQ(list.front().addr(), block1.addr());

  list.pop(get_test_secrets());
  ASSERT_FALSE(list.empty());
  EXPECT_EQ(list.front().addr(), block2.addr());

  list.pop(get_test_secrets());
  ASSERT_TRUE(list.empty());

  list.push(block1, get_test_secrets());
  list.push(block2, get_test_secrets());
  list.remove(reinterpret_cast<FreeList::Node *>(block2.usable_space()),
              get_test_secrets());
  EXPECT_EQ(list.front().addr(), block1.addr());
  list.pop(get_test_secrets());
  ASSERT_TRUE(list.empty());
}

#if LIBC_COPT_HARDEN_FREELIST
TEST(LlvmLibcFreeList, HardenedCorruptNext) {
  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block1 = *maybeBlock;

  maybeBlock = block1.split(128);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block2 = *maybeBlock;

  FreeList list;
  list.push(block1, get_test_secrets());
  list.push(block2, get_test_secrets());

  struct RawNode {
    void *prev;
    void *next;
  };
  RawNode *raw_node2 = reinterpret_cast<RawNode *>(block2.usable_space());
  raw_node2->next = reinterpret_cast<void *>(0xDEADBEEF); // Corrupt next

  EXPECT_DEATH(
      [&] {
        list.pop(get_test_secrets()); // Should trap due to corrupted block2->next
      },
      WITH_SIGNAL(-1));
}

TEST(LlvmLibcFreeList, HardenedCorruptPrev) {
  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block1 = *maybeBlock;

  maybeBlock = block1.split(128);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef block2 = *maybeBlock;

  FreeList list;
  list.push(block1, get_test_secrets());
  list.push(block2, get_test_secrets());

  struct RawNode {
    void *prev;
    void *next;
  };
  RawNode *raw_node2 = reinterpret_cast<RawNode *>(block2.usable_space());
  raw_node2->prev = reinterpret_cast<void *>(0xDEADBEEF); // Corrupt prev

  EXPECT_DEATH(
      [&] {
        list.pop(get_test_secrets()); // Should trap due to corrupted block2->prev
      },
      WITH_SIGNAL(-1));
}
#endif
