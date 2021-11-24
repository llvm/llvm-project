//===-- sanitizer_stack_store_test.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_stack_store.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "sanitizer_hash.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

class StackStoreTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override { store_.TestOnlyUnmap(); }

  template <typename Fn>
  void ForEachTrace(Fn fn, uptr n = 1000000) {
    std::vector<uptr> frames(kStackTraceMax);
    std::iota(frames.begin(), frames.end(), 1);
    MurMur2HashBuilder h(0);
    for (uptr i = 0; i < n; ++i) {
      h.add(i);
      u32 size = h.get() % kStackTraceMax;
      h.add(i);
      uptr tag = h.get() % 256;
      StackTrace s(frames.data(), size, tag);
      if (!s.size && !s.tag)
        continue;
      fn(s);
      std::next_permutation(frames.begin(), frames.end());
    };
  }

  StackStore store_ = {};
};

TEST_F(StackStoreTest, Empty) {
  uptr before = store_.Allocated();
  EXPECT_EQ(0u, store_.Store({}));
  uptr after = store_.Allocated();
  EXPECT_EQ(before, after);
}

TEST_F(StackStoreTest, Basic) {
  std::vector<StackStore::Id> ids;
  ForEachTrace([&](const StackTrace& s) { ids.push_back(store_.Store(s)); });

  auto id = ids.begin();
  ForEachTrace([&](const StackTrace& s) {
    StackTrace trace = store_.Load(*(id++));
    EXPECT_EQ(s.size, trace.size);
    EXPECT_EQ(s.tag, trace.tag);
    EXPECT_EQ(std::vector<uptr>(s.trace, s.trace + s.size),
              std::vector<uptr>(trace.trace, trace.trace + trace.size));
  });
}

TEST_F(StackStoreTest, Allocated) {
  EXPECT_LE(store_.Allocated(), 0x100000u);
  std::vector<StackStore::Id> ids;
  ForEachTrace([&](const StackTrace& s) { ids.push_back(store_.Store(s)); });
  EXPECT_NEAR(store_.Allocated(), FIRST_32_SECOND_64(500000000u, 1000000000u),
              FIRST_32_SECOND_64(50000000u, 100000000u));
  store_.TestOnlyUnmap();
  EXPECT_LE(store_.Allocated(), 0x100000u);
}

}  // namespace __sanitizer
