//===-- Unittests for BlockStore ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/blockstore.h"
#include "utils/UnitTest/Test.h"

struct Element {
  int a;
  long b;
  unsigned c;
};

class LlvmLibcBlockStoreTest : public __llvm_libc::testing::Test {
public:
  template <size_t BLOCK_SIZE, size_t ELEMENT_COUNT, bool REVERSE>
  void populate_and_iterate() {
    __llvm_libc::cpp::BlockStore<Element, BLOCK_SIZE, REVERSE> block_store;
    for (int i = 0; i < int(ELEMENT_COUNT); ++i)
      block_store.push_back({i, 2 * i, 3 * unsigned(i)});
    auto end = block_store.end();
    int i = 0;
    for (auto iter = block_store.begin(); iter != end; ++iter, ++i) {
      Element &e = *iter;
      if (REVERSE) {
        int j = ELEMENT_COUNT - 1 - i;
        ASSERT_EQ(e.a, j);
        ASSERT_EQ(e.b, long(j * 2));
        ASSERT_EQ(e.c, unsigned(j * 3));
      } else {
        ASSERT_EQ(e.a, i);
        ASSERT_EQ(e.b, long(i * 2));
        ASSERT_EQ(e.c, unsigned(i * 3));
      }
    }
    ASSERT_EQ(i, int(ELEMENT_COUNT));
    __llvm_libc::cpp::BlockStore<Element, BLOCK_SIZE, REVERSE>::destroy(
        &block_store);
  }

  template <bool REVERSE> void back_test() {
    using __llvm_libc::cpp::BlockStore;
    BlockStore<int, 4, REVERSE> block_store;
    for (int i = 0; i < 20; i++)
      block_store.push_back(i);
    for (int i = 19; i >= 0; i--, block_store.pop_back())
      ASSERT_EQ(block_store.back(), i);
    block_store.destroy(&block_store);
  }

  template <bool REVERSE> void empty_test() {
    using __llvm_libc::cpp::BlockStore;
    BlockStore<int, 2, REVERSE> block_store;

    ASSERT_TRUE(block_store.empty());
    block_store.push_back(1);
    for (int i = 0; i < 10; i++, block_store.push_back(1))
      ASSERT_FALSE(block_store.empty());
    block_store.destroy(&block_store);
  }
};

TEST_F(LlvmLibcBlockStoreTest, PopulateAndIterate4) {
  populate_and_iterate<4, 4, false>();
}

TEST_F(LlvmLibcBlockStoreTest, PopulateAndIterate8) {
  populate_and_iterate<4, 8, false>();
}

TEST_F(LlvmLibcBlockStoreTest, PopulateAndIterate10) {
  populate_and_iterate<4, 10, false>();
}

TEST_F(LlvmLibcBlockStoreTest, PopulateAndIterateReverse4) {
  populate_and_iterate<4, 4, true>();
}

TEST_F(LlvmLibcBlockStoreTest, PopulateAndIterateReverse8) {
  populate_and_iterate<4, 8, true>();
}

TEST_F(LlvmLibcBlockStoreTest, PopulateAndIterateReverse10) {
  populate_and_iterate<4, 10, true>();
}

TEST_F(LlvmLibcBlockStoreTest, Back) {
  back_test<false>();
  back_test<true>();
}

TEST_F(LlvmLibcBlockStoreTest, Empty) {
  empty_test<false>();
  empty_test<true>();
}
