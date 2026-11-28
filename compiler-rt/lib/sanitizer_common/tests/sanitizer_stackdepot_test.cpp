//===-- sanitizer_stackdepot_test.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_stackdepot.h"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_stackdepotbase.h"

namespace __sanitizer {

class StackDepotTest : public testing::Test {
 protected:
  void SetUp() override { StackDepotTestOnlyUnmap(); }
  void TearDown() override {
    StackDepotStats stack_depot_stats = StackDepotGetStats();
    Printf("StackDepot: %zd ids; %zdM allocated\n",
           stack_depot_stats.n_uniq_ids, stack_depot_stats.allocated >> 20);
    StackDepotTestOnlyUnmap();
  }
};

TEST_F(StackDepotTest, Basic) {
  uptr array[] = {1, 2, 3, 4, 5};
  StackTrace s1(array, ARRAY_SIZE(array));
  u32 i1 = StackDepotPut(s1);
  StackTrace stack = StackDepotGet(i1);
  EXPECT_NE(stack.trace, (uptr*)0);
  EXPECT_EQ(ARRAY_SIZE(array), stack.size);
  EXPECT_EQ(0, internal_memcmp(stack.trace, array, sizeof(array)));
}

TEST_F(StackDepotTest, Absent) {
  StackTrace stack = StackDepotGet((1 << 30) - 1);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST_F(StackDepotTest, EmptyStack) {
  u32 i1 = StackDepotPut(StackTrace());
  StackTrace stack = StackDepotGet(i1);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST_F(StackDepotTest, ZeroId) {
  StackTrace stack = StackDepotGet(0);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST_F(StackDepotTest, Same) {
  uptr array[] = {1, 2, 3, 4, 6};
  StackTrace s1(array, ARRAY_SIZE(array));
  u32 i1 = StackDepotPut(s1);
  u32 i2 = StackDepotPut(s1);
  EXPECT_EQ(i1, i2);
  StackTrace stack = StackDepotGet(i1);
  EXPECT_NE(stack.trace, (uptr*)0);
  EXPECT_EQ(ARRAY_SIZE(array), stack.size);
  EXPECT_EQ(0, internal_memcmp(stack.trace, array, sizeof(array)));
}

TEST_F(StackDepotTest, Several) {
  uptr array1[] = {1, 2, 3, 4, 7};
  StackTrace s1(array1, ARRAY_SIZE(array1));
  u32 i1 = StackDepotPut(s1);
  uptr array2[] = {1, 2, 3, 4, 8, 9};
  StackTrace s2(array2, ARRAY_SIZE(array2));
  u32 i2 = StackDepotPut(s2);
  EXPECT_NE(i1, i2);
}

TEST_F(StackDepotTest, Print) {
  uptr array1[] = {0x111, 0x222, 0x333, 0x444, 0x777};
  StackTrace s1(array1, ARRAY_SIZE(array1));
  u32 i1 = StackDepotPut(s1);
  uptr array2[] = {0x1111, 0x2222, 0x3333, 0x4444, 0x8888, 0x9999};
  StackTrace s2(array2, ARRAY_SIZE(array2));
  u32 i2 = StackDepotPut(s2);
  EXPECT_NE(i1, i2);

  auto fix_regex = [](const std::string& s) -> std::string {
    if (!SANITIZER_WINDOWS)
      return s;
    return std::regex_replace(s, std::regex("\\.\\*"), ".*\\n.*");
  };
  EXPECT_EXIT(
      (StackDepotPrintAll(), exit(0)), ::testing::ExitedWithCode(0),
      fix_regex(
          "Stack for id .*#0 0x0*1.*#1 0x0*2.*#2 0x0*3.*#3 0x0*4.*#4 0x0*7.*"));
  EXPECT_EXIT((StackDepotPrintAll(), exit(0)), ::testing::ExitedWithCode(0),
              fix_regex("Stack for id .*#0 0x0*1.*#1 0x0*2.*#2 0x0*3.*#3 "
                        "0x0*4.*#4 0x0*8.*#5 0x0*9.*"));
}

TEST_F(StackDepotTest, PrintNoLock) {
  u32 n = 2000;
  std::vector<u32> idx2id(n);
  for (u32 i = 0; i < n; ++i) {
    uptr array[] = {0x111, 0x222, i, 0x444, 0x777};
    StackTrace s(array, ARRAY_SIZE(array));
    idx2id[i] = StackDepotPut(s);
  }
  StackDepotPrintAll();
  for (u32 i = 0; i < n; ++i) {
    uptr array[] = {0x111, 0x222, i, 0x444, 0x777};
    StackTrace s(array, ARRAY_SIZE(array));
    CHECK_EQ(idx2id[i], StackDepotPut(s));
  }
}

// All values share bucket 0. While `hold` is set, store() does not return, so
// that a test knows that a thread holds the lock of that bucket.
struct OneBucketNode {
  using args_type = u32;
  using hash_type = u32;
  using handle_type = void;
  using Depot = StackDepotBase<OneBucketNode, 1, 4>;

  u32 value;
  u32 link;

  static hash_type hash(u32) { return 0; }
  static bool is_valid(u32 v) { return v; }
  static uptr allocated() { return 0; }
  bool eq(hash_type, u32 v) const { return v == value; }
  u32 load(u32) const { return value; }
  void store(u32, u32 v, hash_type) {
    value = v;
    in_store = true;
    while (hold) std::this_thread::yield();
  }

  static std::atomic<bool> hold;
  static std::atomic<bool> in_store;

  // The node type is a friend of the depot.
  static bool IsLocked(const Depot& depot, uptr bucket);
  static void Lock(Depot& depot, uptr bucket);
  static u32 Lockers(const Depot& depot);
};

std::atomic<bool> OneBucketNode::hold;
std::atomic<bool> OneBucketNode::in_store;

bool OneBucketNode::IsLocked(const Depot& depot, uptr bucket) {
  return atomic_load_relaxed(&depot.tab[bucket]) & Depot::kLockMask;
}

void OneBucketNode::Lock(Depot& depot, uptr bucket) {
  Depot::lock(&depot.tab[bucket]);
}

u32 OneBucketNode::Lockers(const Depot& depot) {
  return atomic_load_relaxed(&depot.n_lockers);
}

// These tests call the fork hooks as a child would, without forking.
class StackDepotBaseForkTest : public testing::Test {
 protected:
  void TearDown() override { depot.TestOnlyUnmap(); }

  std::thread PutAndHoldLock(u32 value) {
    OneBucketNode::in_store = false;
    OneBucketNode::hold = true;
    std::thread thread([value] { depot.Put(value); });
    while (!OneBucketNode::in_store) std::this_thread::yield();
    return thread;
  }

  void RunForkHooksOfChild() {
    depot.LockBeforeFork();
    depot.UnlockAfterFork(/*fork_child=*/true);
  }

  static OneBucketNode::Depot depot;
};

OneBucketNode::Depot StackDepotBaseForkTest::depot;

TEST_F(StackDepotBaseForkTest, UnlocksBucketOfInsertionInProgress) {
  std::thread thread = PutAndHoldLock(1);
  EXPECT_TRUE(OneBucketNode::IsLocked(depot, 0));
  RunForkHooksOfChild();
  EXPECT_FALSE(OneBucketNode::IsLocked(depot, 0));
  // The interrupted Put may go on in the child (fork from a signal handler).
  OneBucketNode::hold = false;
  thread.join();
  EXPECT_EQ(0u, OneBucketNode::Lockers(depot));
}

#if SANITIZER_LINUX
TEST_F(StackDepotBaseForkTest, SkipsTableIfNoInsertionInProgress) {
  // The second thread waits for the lock and then finds the value, so both ways
  // out of the locked part of Put are taken.
  std::thread inserts = PutAndHoldLock(1);
  std::thread finds([] { depot.Put(1); });
  while (OneBucketNode::Lockers(depot) != 2) std::this_thread::yield();
  OneBucketNode::hold = false;
  inserts.join();
  finds.join();
  EXPECT_EQ(1u, depot.GetStats().n_uniq_ids);
  // A lock that no Put has taken shows whether the table is walked.
  OneBucketNode::Lock(depot, 1);
  RunForkHooksOfChild();
  EXPECT_TRUE(OneBucketNode::IsLocked(depot, 1));
}
#endif  // SANITIZER_LINUX

static struct StackDepotBenchmarkParams {
  int UniqueStacksPerThread;
  int RepeatPerThread;
  int Threads;
  bool UniqueThreads;
  bool UseCount;
} params[] = {
    // All traces are unique, very unusual.
    {10000000, 1, 1, false, false},
    {8000000, 1, 4, false, false},
    {8000000, 1, 16, false, false},
    // Probably most realistic sets.
    {3000000, 10, 1, false, false},
    {3000000, 10, 4, false, false},
    {3000000, 10, 16, false, false},
    // Update use count as msan/dfsan.
    {3000000, 10, 1, false, true},
    {3000000, 10, 4, false, true},
    {3000000, 10, 16, false, true},
    // Unrealistic, as above, but traces are unique inside of thread.
    {4000000, 1, 4, true, false},
    {2000000, 1, 16, true, false},
    {2000000, 10, 4, true, false},
    {500000, 10, 16, true, false},
    {1500000, 10, 4, true, true},
    {800000, 10, 16, true, true},
    // Go crazy, and create too many unique stacks, such that StackStore runs
    // out of space.
    {1000000, 1, 128, true, true},
    {100000000, 1, 1, true, true},
};

static std::string PrintStackDepotBenchmarkParams(
    const testing::TestParamInfo<StackDepotBenchmarkParams>& info) {
  std::stringstream name;
  name << info.param.UniqueStacksPerThread << "_" << info.param.RepeatPerThread
       << "_" << info.param.Threads << (info.param.UseCount ? "_UseCount" : "")
       << (info.param.UniqueThreads ? "_UniqueThreads" : "");
  return name.str();
}

class StackDepotBenchmark
    : public StackDepotTest,
      public testing::WithParamInterface<StackDepotBenchmarkParams> {};

// Test which can be used as a simple benchmark. It's disabled to avoid slowing
// down check-sanitizer.
// Usage: Sanitizer-<ARCH>-Test --gtest_also_run_disabled_tests \
//   '--gtest_filter=*Benchmark*'
TEST_P(StackDepotBenchmark, DISABLED_Benchmark) {
  auto Param = GetParam();
  std::atomic<int> here = {};

  auto thread = [&](int idx) {
    here++;
    while (here < Param.Threads) std::this_thread::yield();

    std::vector<uptr> frames(64);
    for (int r = 0; r < Param.RepeatPerThread; ++r) {
      std::iota(frames.begin(), frames.end(), idx + 1);
      for (int i = 0; i < Param.UniqueStacksPerThread; ++i) {
        StackTrace s(frames.data(), frames.size());
        auto h = StackDepotPut_WithHandle(s);
        if (Param.UseCount)
          h.inc_use_count_unsafe();
        std::next_permutation(frames.begin(), frames.end());
      };
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < Param.Threads; ++i)
    threads.emplace_back(thread, Param.UniqueThreads * i);
  for (auto& t : threads) t.join();
}

INSTANTIATE_TEST_SUITE_P(StackDepotBenchmarkSuite, StackDepotBenchmark,
                         testing::ValuesIn(params),
                         PrintStackDepotBenchmarkParams);

}  // namespace __sanitizer
