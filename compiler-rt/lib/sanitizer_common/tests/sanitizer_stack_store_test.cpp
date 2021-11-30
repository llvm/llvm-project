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
#include "sanitizer_atomic.h"
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

  using BlockInfo = StackStore::BlockInfo;

  uptr GetTotalFramesCount() const {
    return atomic_load_relaxed(&store_.total_frames_);
  }

  uptr CountReadyToPackBlocks() {
    uptr res = 0;
    for (BlockInfo& b : store_.blocks_) res += b.Stored(0);
    return res;
  }

  uptr IdToOffset(StackStore::Id id) const { return store_.IdToOffset(id); }

  static constexpr uptr kBlockSizeFrames = StackStore::kBlockSizeFrames;

  StackStore store_ = {};
};

TEST_F(StackStoreTest, Empty) {
  uptr before = store_.Allocated();
  uptr pack = 0;
  EXPECT_EQ(0u, store_.Store({}, &pack));
  uptr after = store_.Allocated();
  EXPECT_EQ(before, after);
}

TEST_F(StackStoreTest, Basic) {
  std::vector<StackStore::Id> ids;
  ForEachTrace([&](const StackTrace& s) {
    uptr pack = 0;
    ids.push_back(store_.Store(s, &pack));
  });

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
  ForEachTrace([&](const StackTrace& s) {
    uptr pack = 0;
    ids.push_back(store_.Store(s, &pack));
  });
  EXPECT_NEAR(store_.Allocated(), FIRST_32_SECOND_64(500000000u, 1000000000u),
              FIRST_32_SECOND_64(50000000u, 100000000u));
  store_.TestOnlyUnmap();
  EXPECT_LE(store_.Allocated(), 0x100000u);
}

TEST_F(StackStoreTest, ReadyToPack) {
  uptr next_pack = kBlockSizeFrames;
  uptr total_ready = 0;
  ForEachTrace(
      [&](const StackTrace& s) {
        uptr pack = 0;
        StackStore::Id id = store_.Store(s, &pack);
        uptr end_idx = IdToOffset(id) + 1 + s.size;
        if (end_idx >= next_pack) {
          EXPECT_EQ(1u, pack);
          next_pack += kBlockSizeFrames;
        } else {
          EXPECT_EQ(0u, pack);
        }
        total_ready += pack;
        EXPECT_EQ(CountReadyToPackBlocks(), total_ready);
      },
      100000);
  EXPECT_EQ(GetTotalFramesCount() / kBlockSizeFrames, total_ready);
}

}  // namespace __sanitizer
