//===-- Unittests for BlockStore ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/blockstore.h"
#include "test/UnitTest/Test.h"

struct Element {
  int a;
  long b;
  unsigned c;
};

class LlvmLibcBlockStoreTest : public LIBC_NAMESPACE::testing::Test {
public:
  template <size_t BLOCK_SIZE, size_t ELEMENT_COUNT, bool REVERSE>
  void populate_and_iterate() {
    LIBC_NAMESPACE::BlockStore<Element, BLOCK_SIZE, REVERSE> block_store;
    for (int i = 0; i < int(ELEMENT_COUNT); ++i)
      ASSERT_TRUE(block_store.push_back({i, 2 * i, 3 * unsigned(i)}));
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
    LIBC_NAMESPACE::BlockStore<Element, BLOCK_SIZE, REVERSE>::destroy(
        &block_store);
  }

  template <bool REVERSE> void back_test() {
    using LIBC_NAMESPACE::BlockStore;
    BlockStore<int, 4, REVERSE> block_store;
    for (int i = 0; i < 20; i++)
      ASSERT_TRUE(block_store.push_back(i));
    for (int i = 19; i >= 0; i--, block_store.pop_back())
      ASSERT_EQ(block_store.back(), i);
    block_store.destroy(&block_store);
  }

  template <bool REVERSE> void empty_test() {
    using LIBC_NAMESPACE::BlockStore;
    BlockStore<int, 2, REVERSE> block_store;

    ASSERT_TRUE(block_store.empty());
    ASSERT_TRUE(block_store.push_back(1));
    for (int i = 0; i < 10; i++) {
      ASSERT_FALSE(block_store.empty());
      ASSERT_TRUE(block_store.push_back(1));
    }
    block_store.destroy(&block_store);
  }

  template <bool REVERSE> void erase_test() {
    using LIBC_NAMESPACE::BlockStore;
    BlockStore<int, 2, REVERSE> block_store;
    int i;

    constexpr int ARR_SIZE = 6;

    ASSERT_TRUE(block_store.empty());
    for (int i = 0; i < ARR_SIZE; i++) {
      ASSERT_TRUE(block_store.push_back(i + 1));
    }

    // block_store state should be {1,2,3,4,5,6}

    block_store.erase(block_store.begin());

    // FORWARD: block_store state should be {2,3,4,5,6}
    // REVERSE: block_store state should be {1,2,3,4,5}

    auto iter = block_store.begin();
    for (i = 0; iter != block_store.end(); ++i, ++iter) {
      if (!REVERSE) {
        ASSERT_EQ(*iter, i + 2);
      } else {
        ASSERT_EQ(*iter, (ARR_SIZE - 1) - i);
      }
    }

    // Assert that there were the correct number of elements
    ASSERT_EQ(i, ARR_SIZE - 1);

    block_store.erase(block_store.end());

    // BOTH: block_store state should be {2,3,4,5}

    iter = block_store.begin();
    for (i = 0; iter != block_store.end(); ++i, ++iter) {
      if (!REVERSE) {
        ASSERT_EQ(*iter, i + 2);
      } else {
        ASSERT_EQ(*iter, (ARR_SIZE - 1) - i);
      }
    }

    ASSERT_EQ(i, ARR_SIZE - 2);

    block_store.erase(block_store.begin() + 1);

    // FORWARD: block_store state should be {2,4,5}
    // REVERSE: block_store state should be {2,3,5}

    const int FORWARD_RESULTS[] = {2, 4, 5};
    const int REVERSE_RESULTS[] = {2, 3, 5};

    iter = block_store.begin();
    for (i = 0; iter != block_store.end(); ++i, ++iter) {
      if (!REVERSE) {
        ASSERT_EQ(*iter, FORWARD_RESULTS[i]);
      } else {
        ASSERT_EQ(*iter, REVERSE_RESULTS[ARR_SIZE - 4 - i]); // reversed
      }
    }

    ASSERT_EQ(i, ARR_SIZE - 3);

    block_store.erase(block_store.begin() + 1);
    // BOTH: block_store state should be {2,5}

    iter = block_store.begin();
    if (!REVERSE) {
      ASSERT_EQ(*iter, 2);
      ASSERT_EQ(*(iter + 1), 5);
    } else {
      ASSERT_EQ(*iter, 5);
      ASSERT_EQ(*(iter + 1), 2);
    }

    block_store.erase(block_store.begin());
    // FORWARD: block_store state should be {5}
    // REVERSE: block_store state should be {2}
    iter = block_store.begin();
    if (!REVERSE) {
      ASSERT_EQ(*iter, 5);
    } else {
      ASSERT_EQ(*iter, 2);
    }

    block_store.erase(block_store.begin());
    // BOTH: block_store state should be {}

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

TEST_F(LlvmLibcBlockStoreTest, Erase) {
  erase_test<false>();
  erase_test<true>();
}
