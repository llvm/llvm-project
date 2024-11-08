//===-- Unittests for table -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h" // bit_ceil
#include "src/__support/HashTable/randomness.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {
TEST(LlvmLibcTableTest, AllocationAndDeallocation) {
  size_t caps[] = {0, 1, 2, 3, 4, 7, 11, 37, 1024, 5261, 19999};
  const char *keys[] = {"",         "a",         "ab",        "abc",
                        "abcd",     "abcde",     "abcdef",    "abcdefg",
                        "abcdefgh", "abcdefghi", "abcdefghij"};
  for (size_t i : caps) {
    HashTable *table = HashTable::allocate(i, 1);
    ASSERT_NE(table, static_cast<HashTable *>(nullptr));
    for (const char *key : keys) {
      ASSERT_EQ(table->find(key), static_cast<ENTRY *>(nullptr));
    }
    HashTable::deallocate(table);
  }
  ASSERT_EQ(HashTable::allocate(-1, 0), static_cast<HashTable *>(nullptr));
  HashTable::deallocate(nullptr);
}

TEST(LlvmLibcTableTest, Iteration) {
  constexpr size_t TEST_SIZE = 512;
  size_t counter[TEST_SIZE];
  struct key {
    uint8_t bytes[3];
  } keys[TEST_SIZE];
  HashTable *table = HashTable::allocate(0, 0x7f7f7f7f7f7f7f7f);
  ASSERT_NE(table, static_cast<HashTable *>(nullptr));
  for (size_t i = 0; i < TEST_SIZE; ++i) {
    counter[i] = 0;
    if (i >= 256) {
      keys[i].bytes[0] = 2;
      keys[i].bytes[1] = i % 256;
      keys[i].bytes[2] = 0;
    } else {
      keys[i].bytes[0] = 1;
      keys[i].bytes[1] = i;
      keys[i].bytes[2] = 0;
    }
    HashTable::insert(table, {reinterpret_cast<char *>(keys[i].bytes),
                              reinterpret_cast<void *>((size_t)i)});
  }

  size_t count = 0;
  for (const ENTRY &e : *table) {
    size_t data = reinterpret_cast<size_t>(e.data);
    ++counter[data];
    ++count;
  }
  ASSERT_EQ(count, TEST_SIZE);
  for (size_t i = 0; i < TEST_SIZE; ++i) {
    ASSERT_EQ(counter[i], static_cast<size_t>(1));
  }
  HashTable::deallocate(table);
}

// Check if resize works correctly. This test actually covers two things:
// - The sizes are indeed growing.
// - The sizes are growing rapidly enough to reach the upper bound.
TEST(LlvmLibcTableTest, GrowthSequence) {
  size_t cap = capacity_to_entries(0);
  // right shift 4 to avoid overflow ssize_t.
  while (cap < static_cast<size_t>(-1) >> 4u) {
    size_t hint = cap / 8 * 7 + 1;
    size_t new_cap = capacity_to_entries(hint);
    ASSERT_GT(new_cap, cap);
    cap = new_cap;
  }
}

TEST(LlvmLibcTableTest, Insertion) {
  union key {
    char bytes[2];
  } keys[256];
  for (size_t k = 0; k < 256; ++k) {
    keys[k].bytes[0] = static_cast<char>(k);
    keys[k].bytes[1] = 0;
  }
  constexpr size_t CAP = cpp::bit_ceil((sizeof(Group) + 1) * 8 / 7) / 8 * 7;
  static_assert(CAP + 1 < 256, "CAP is too large for this test.");
  HashTable *table =
      HashTable::allocate(sizeof(Group) + 1, randomness::next_random_seed());
  ASSERT_NE(table, static_cast<HashTable *>(nullptr));

  // insert to full capacity.
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_NE(HashTable::insert(table, {keys[i].bytes, keys[i].bytes}),
              static_cast<ENTRY *>(nullptr));
  }

  // One more insert should grow the table successfully. We test the value
  // here because the grow finishes with a fastpath insertion that is different
  // from the normal insertion.
  ASSERT_EQ(HashTable::insert(table, {keys[CAP].bytes, keys[CAP].bytes})->data,
            static_cast<void *>(keys[CAP].bytes));

  for (size_t i = 0; i <= CAP; ++i) {
    ASSERT_EQ(strcmp(table->find(keys[i].bytes)->key, keys[i].bytes), 0);
  }
  for (size_t i = CAP + 1; i < 256; ++i) {
    ASSERT_EQ(table->find(keys[i].bytes), static_cast<ENTRY *>(nullptr));
  }

  // do not replace old value
  for (size_t i = 0; i <= CAP; ++i) {
    ASSERT_NE(
        HashTable::insert(table, {keys[i].bytes, reinterpret_cast<void *>(i)}),
        static_cast<ENTRY *>(nullptr));
  }
  for (size_t i = 0; i <= CAP; ++i) {
    ASSERT_EQ(table->find(keys[i].bytes)->data,
              reinterpret_cast<void *>(keys[i].bytes));
  }

  HashTable::deallocate(table);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
