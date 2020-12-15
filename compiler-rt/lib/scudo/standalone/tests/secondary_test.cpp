//===-- secondary_test.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "allocator_config.h"
#include "secondary.h"

#include <stdio.h>

#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

template <class SecondaryT> static void testSecondaryBasic(void) {
  scudo::GlobalStats S;
  S.init();
  std::unique_ptr<SecondaryT> L(new SecondaryT);
  L->init(&S);
  const scudo::uptr Size = 1U << 16;
  void *P = L->allocate(Size);
  EXPECT_NE(P, nullptr);
  memset(P, 'A', Size);
  EXPECT_GE(SecondaryT::getBlockSize(P), Size);
  L->deallocate(P);
  // If the Secondary can't cache that pointer, it will be unmapped.
  if (!L->canCache(Size))
    EXPECT_DEATH(memset(P, 'A', Size), "");

  const scudo::uptr Align = 1U << 16;
  P = L->allocate(Size + Align, Align);
  EXPECT_NE(P, nullptr);
  void *AlignedP = reinterpret_cast<void *>(
      scudo::roundUpTo(reinterpret_cast<scudo::uptr>(P), Align));
  memset(AlignedP, 'A', Size);
  L->deallocate(P);

  std::vector<void *> V;
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(L->allocate(Size));
  std::shuffle(V.begin(), V.end(), std::mt19937(std::random_device()()));
  while (!V.empty()) {
    L->deallocate(V.back());
    V.pop_back();
  }
  scudo::ScopedString Str(1024);
  L->getStats(&Str);
  Str.output();
}

struct TestConfig {
  static const scudo::u32 SecondaryCacheEntriesArraySize = 128U;
  static const scudo::u32 SecondaryCacheDefaultMaxEntriesCount = 64U;
  static const scudo::uptr SecondaryCacheDefaultMaxEntrySize = 1UL << 20;
  static const scudo::s32 SecondaryCacheMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 SecondaryCacheMaxReleaseToOsIntervalMs = INT32_MAX;
};

TEST(ScudoSecondaryTest, SecondaryBasic) {
  testSecondaryBasic<scudo::MapAllocator<scudo::MapAllocatorNoCache>>();
  testSecondaryBasic<
      scudo::MapAllocator<scudo::MapAllocatorCache<scudo::DefaultConfig>>>();
  testSecondaryBasic<
      scudo::MapAllocator<scudo::MapAllocatorCache<TestConfig>>>();
}

using LargeAllocator =
    scudo::MapAllocator<scudo::MapAllocatorCache<scudo::DefaultConfig>>;

// This exercises a variety of combinations of size and alignment for the
// MapAllocator. The size computation done here mimic the ones done by the
// combined allocator.
TEST(ScudoSecondaryTest, SecondaryCombinations) {
  constexpr scudo::uptr MinAlign = FIRST_32_SECOND_64(8, 16);
  constexpr scudo::uptr HeaderSize = scudo::roundUpTo(8, MinAlign);
  std::unique_ptr<LargeAllocator> L(new LargeAllocator);
  L->init(nullptr);
  for (scudo::uptr SizeLog = 0; SizeLog <= 20; SizeLog++) {
    for (scudo::uptr AlignLog = FIRST_32_SECOND_64(3, 4); AlignLog <= 16;
         AlignLog++) {
      const scudo::uptr Align = 1U << AlignLog;
      for (scudo::sptr Delta = -128; Delta <= 128; Delta += 8) {
        if (static_cast<scudo::sptr>(1U << SizeLog) + Delta <= 0)
          continue;
        const scudo::uptr UserSize =
            scudo::roundUpTo((1U << SizeLog) + Delta, MinAlign);
        const scudo::uptr Size =
            HeaderSize + UserSize + (Align > MinAlign ? Align - HeaderSize : 0);
        void *P = L->allocate(Size, Align);
        EXPECT_NE(P, nullptr);
        void *AlignedP = reinterpret_cast<void *>(
            scudo::roundUpTo(reinterpret_cast<scudo::uptr>(P), Align));
        memset(AlignedP, 0xff, UserSize);
        L->deallocate(P);
      }
    }
  }
  scudo::ScopedString Str(1024);
  L->getStats(&Str);
  Str.output();
}

TEST(ScudoSecondaryTest, SecondaryIterate) {
  std::unique_ptr<LargeAllocator> L(new LargeAllocator);
  L->init(nullptr);
  std::vector<void *> V;
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(L->allocate((std::rand() % 16) * PageSize));
  auto Lambda = [V](scudo::uptr Block) {
    EXPECT_NE(std::find(V.begin(), V.end(), reinterpret_cast<void *>(Block)),
              V.end());
  };
  L->disable();
  L->iterateOverBlocks(Lambda);
  L->enable();
  while (!V.empty()) {
    L->deallocate(V.back());
    V.pop_back();
  }
  scudo::ScopedString Str(1024);
  L->getStats(&Str);
  Str.output();
}

TEST(ScudoSecondaryTest, SecondaryOptions) {
  std::unique_ptr<LargeAllocator> L(new LargeAllocator);
  L->init(nullptr);
  // Attempt to set a maximum number of entries higher than the array size.
  EXPECT_FALSE(L->setOption(scudo::Option::MaxCacheEntriesCount, 4096U));
  // A negative number will be cast to a scudo::u32, and fail.
  EXPECT_FALSE(L->setOption(scudo::Option::MaxCacheEntriesCount, -1));
  if (L->canCache(0U)) {
    // Various valid combinations.
    EXPECT_TRUE(L->setOption(scudo::Option::MaxCacheEntriesCount, 4U));
    EXPECT_TRUE(L->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 20));
    EXPECT_TRUE(L->canCache(1UL << 18));
    EXPECT_TRUE(L->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 17));
    EXPECT_FALSE(L->canCache(1UL << 18));
    EXPECT_TRUE(L->canCache(1UL << 16));
    EXPECT_TRUE(L->setOption(scudo::Option::MaxCacheEntriesCount, 0U));
    EXPECT_FALSE(L->canCache(1UL << 16));
    EXPECT_TRUE(L->setOption(scudo::Option::MaxCacheEntriesCount, 4U));
    EXPECT_TRUE(L->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 20));
    EXPECT_TRUE(L->canCache(1UL << 16));
  }
}

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready;

static void performAllocations(LargeAllocator *L) {
  std::vector<void *> V;
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  for (scudo::uptr I = 0; I < 128U; I++) {
    // Deallocate 75% of the blocks.
    const bool Deallocate = (rand() & 3) != 0;
    void *P = L->allocate((std::rand() % 16) * PageSize);
    if (Deallocate)
      L->deallocate(P);
    else
      V.push_back(P);
  }
  while (!V.empty()) {
    L->deallocate(V.back());
    V.pop_back();
  }
}

TEST(ScudoSecondaryTest, SecondaryThreadsRace) {
  Ready = false;
  std::unique_ptr<LargeAllocator> L(new LargeAllocator);
  L->init(nullptr, /*ReleaseToOsInterval=*/0);
  std::thread Threads[16];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(performAllocations, L.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  scudo::ScopedString Str(1024);
  L->getStats(&Str);
  Str.output();
}
