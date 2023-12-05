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
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {
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

TEST(LlvmLibcTableTest, Insertion) {
  union key {
    uint64_t value;
    char bytes[8];
  } keys[256];
  for (size_t k = 0; k < 256; ++k) {
    keys[k].value = LIBC_NAMESPACE::Endian::to_little_endian(k);
  }
  constexpr size_t CAP = cpp::bit_ceil((sizeof(Group) + 1) * 8 / 7) / 8 * 7;
  static_assert(CAP + 1 < 256, "CAP is too large for this test.");
  HashTable *table =
      HashTable::allocate(sizeof(Group) + 1, randomness::next_random_seed());
  ASSERT_NE(table, static_cast<HashTable *>(nullptr));

  // insert to full capacity.
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_NE(table->insert({keys[i].bytes, keys[i].bytes}),
              static_cast<ENTRY *>(nullptr));
  }

  // one more insert should fail.
  ASSERT_EQ(table->insert({keys[CAP + 1].bytes, keys[CAP + 1].bytes}),
            static_cast<ENTRY *>(nullptr));

  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_EQ(strcmp(table->find(keys[i].bytes)->key, keys[i].bytes), 0);
  }
  for (size_t i = CAP; i < 256; ++i) {
    ASSERT_EQ(table->find(keys[i].bytes), static_cast<ENTRY *>(nullptr));
  }

  // do not replace old value
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_NE(table->insert({keys[i].bytes, reinterpret_cast<void *>(i)}),
              static_cast<ENTRY *>(nullptr));
  }
  for (size_t i = 0; i < CAP; ++i) {
    ASSERT_EQ(table->find(keys[i].bytes)->data,
              reinterpret_cast<void *>(keys[i].bytes));
  }

  HashTable::deallocate(table);
}

} // namespace internal
} // namespace LIBC_NAMESPACE
