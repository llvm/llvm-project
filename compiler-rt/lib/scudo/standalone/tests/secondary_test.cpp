//===-- secondary_test.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memtag.h"
#include "tests/scudo_unit_test.h"

#include "allocator_config.h"
#include "allocator_config_wrapper.h"
#include "secondary.h"

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <stdio.h>
#include <thread>
#include <vector>

template <typename Config> static scudo::Options getOptionsForConfig() {
  if (!Config::getMaySupportMemoryTagging() ||
      !scudo::archSupportsMemoryTagging() ||
      !scudo::systemSupportsMemoryTagging())
    return {};
  scudo::AtomicOptions AO;
  AO.set(scudo::OptionBit::UseMemoryTagging);
  return AO.load();
}

template <typename Config> static void testSecondaryBasic(void) {
  using SecondaryT = scudo::MapAllocator<scudo::SecondaryConfig<Config>>;
  scudo::Options Options =
      getOptionsForConfig<scudo::SecondaryConfig<Config>>();

  scudo::GlobalStats S;
  S.init();
  std::unique_ptr<SecondaryT> L(new SecondaryT);
  L->init(&S);
  const scudo::uptr Size = 1U << 16;
  void *P = L->allocate(Options, Size);
  EXPECT_NE(P, nullptr);
  memset(P, 'A', Size);
  EXPECT_GE(SecondaryT::getBlockSize(P), Size);
  L->deallocate(Options, P);

  // If the Secondary can't cache that pointer, it will be unmapped.
  if (!L->canCache(Size)) {
    EXPECT_DEATH(
        {
          // Repeat few time to avoid missing crash if it's mmaped by unrelated
          // code.
          for (int i = 0; i < 10; ++i) {
            P = L->allocate(Options, Size);
            L->deallocate(Options, P);
            memset(P, 'A', Size);
          }
        },
        "");
  }

  const scudo::uptr Align = 1U << 16;
  P = L->allocate(Options, Size + Align, Align);
  EXPECT_NE(P, nullptr);
  void *AlignedP = reinterpret_cast<void *>(
      scudo::roundUp(reinterpret_cast<scudo::uptr>(P), Align));
  memset(AlignedP, 'A', Size);
  L->deallocate(Options, P);

  std::vector<void *> V;
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(L->allocate(Options, Size));
  std::shuffle(V.begin(), V.end(), std::mt19937(std::random_device()()));
  while (!V.empty()) {
    L->deallocate(Options, V.back());
    V.pop_back();
  }
  scudo::ScopedString Str;
  L->getStats(&Str);
  Str.output();
  L->unmapTestOnly();
}

struct NoCacheConfig {
  static const bool MaySupportMemoryTagging = false;
  template <typename> using TSDRegistryT = void;
  template <typename> using PrimaryT = void;
  template <typename Config> using SecondaryT = scudo::MapAllocator<Config>;

  struct Secondary {
    template <typename Config>
    using CacheT = scudo::MapAllocatorNoCache<Config>;
  };
};

struct TestConfig {
  static const bool MaySupportMemoryTagging = false;
  template <typename> using TSDRegistryT = void;
  template <typename> using PrimaryT = void;
  template <typename> using SecondaryT = void;

  struct Secondary {
    struct Cache {
      static const scudo::u32 EntriesArraySize = 128U;
      static const scudo::u32 QuarantineSize = 0U;
      static const scudo::u32 DefaultMaxEntriesCount = 64U;
      static const scudo::uptr DefaultMaxEntrySize = 1UL << 20;
      static const scudo::s32 MinReleaseToOsIntervalMs = INT32_MIN;
      static const scudo::s32 MaxReleaseToOsIntervalMs = INT32_MAX;
    };

    template <typename Config> using CacheT = scudo::MapAllocatorCache<Config>;
  };
};

TEST(ScudoSecondaryTest, SecondaryBasic) {
  testSecondaryBasic<NoCacheConfig>();
  testSecondaryBasic<scudo::DefaultConfig>();
  testSecondaryBasic<TestConfig>();
}

struct MapAllocatorTest : public Test {
  using Config = scudo::DefaultConfig;
  using LargeAllocator = scudo::MapAllocator<scudo::SecondaryConfig<Config>>;

  void SetUp() override { Allocator->init(nullptr); }

  void TearDown() override { Allocator->unmapTestOnly(); }

  std::unique_ptr<LargeAllocator> Allocator =
      std::make_unique<LargeAllocator>();
  scudo::Options Options =
      getOptionsForConfig<scudo::SecondaryConfig<Config>>();
};

// This exercises a variety of combinations of size and alignment for the
// MapAllocator. The size computation done here mimic the ones done by the
// combined allocator.
TEST_F(MapAllocatorTest, SecondaryCombinations) {
  constexpr scudo::uptr MinAlign = FIRST_32_SECOND_64(8, 16);
  constexpr scudo::uptr HeaderSize = scudo::roundUp(8, MinAlign);
  for (scudo::uptr SizeLog = 0; SizeLog <= 20; SizeLog++) {
    for (scudo::uptr AlignLog = FIRST_32_SECOND_64(3, 4); AlignLog <= 16;
         AlignLog++) {
      const scudo::uptr Align = 1U << AlignLog;
      for (scudo::sptr Delta = -128; Delta <= 128; Delta += 8) {
        if ((1LL << SizeLog) + Delta <= 0)
          continue;
        const scudo::uptr UserSize = scudo::roundUp(
            static_cast<scudo::uptr>((1LL << SizeLog) + Delta), MinAlign);
        const scudo::uptr Size =
            HeaderSize + UserSize + (Align > MinAlign ? Align - HeaderSize : 0);
        void *P = Allocator->allocate(Options, Size, Align);
        EXPECT_NE(P, nullptr);
        void *AlignedP = reinterpret_cast<void *>(
            scudo::roundUp(reinterpret_cast<scudo::uptr>(P), Align));
        memset(AlignedP, 0xff, UserSize);
        Allocator->deallocate(Options, P);
      }
    }
  }
  scudo::ScopedString Str;
  Allocator->getStats(&Str);
  Str.output();
}

TEST_F(MapAllocatorTest, SecondaryIterate) {
  std::vector<void *> V;
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(Allocator->allocate(
        Options, (static_cast<scudo::uptr>(std::rand()) % 16U) * PageSize));
  auto Lambda = [&V](scudo::uptr Block) {
    EXPECT_NE(std::find(V.begin(), V.end(), reinterpret_cast<void *>(Block)),
              V.end());
  };
  Allocator->disable();
  Allocator->iterateOverBlocks(Lambda);
  Allocator->enable();
  while (!V.empty()) {
    Allocator->deallocate(Options, V.back());
    V.pop_back();
  }
  scudo::ScopedString Str;
  Allocator->getStats(&Str);
  Str.output();
}

TEST_F(MapAllocatorTest, SecondaryCacheOptions) {
  if (!Allocator->canCache(0U))
    TEST_SKIP("Secondary Cache disabled");

  // Attempt to set a maximum number of entries higher than the array size.
  EXPECT_TRUE(Allocator->setOption(scudo::Option::MaxCacheEntriesCount, 4096U));

  // Attempt to set an invalid (negative) number of entries
  EXPECT_FALSE(Allocator->setOption(scudo::Option::MaxCacheEntriesCount, -1));

  // Various valid combinations.
  EXPECT_TRUE(Allocator->setOption(scudo::Option::MaxCacheEntriesCount, 4U));
  EXPECT_TRUE(
      Allocator->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 20));
  EXPECT_TRUE(Allocator->canCache(1UL << 18));
  EXPECT_TRUE(
      Allocator->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 17));
  EXPECT_FALSE(Allocator->canCache(1UL << 18));
  EXPECT_TRUE(Allocator->canCache(1UL << 16));
  EXPECT_TRUE(Allocator->setOption(scudo::Option::MaxCacheEntriesCount, 0U));
  EXPECT_FALSE(Allocator->canCache(1UL << 16));
  EXPECT_TRUE(Allocator->setOption(scudo::Option::MaxCacheEntriesCount, 4U));
  EXPECT_TRUE(
      Allocator->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 20));
  EXPECT_TRUE(Allocator->canCache(1UL << 16));
}

struct MapAllocatorWithReleaseTest : public MapAllocatorTest {
  void SetUp() override { Allocator->init(nullptr, /*ReleaseToOsInterval=*/0); }

  void performAllocations() {
    std::vector<void *> V;
    const scudo::uptr PageSize = scudo::getPageSizeCached();
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      while (!Ready)
        Cv.wait(Lock);
    }
    for (scudo::uptr I = 0; I < 128U; I++) {
      // Deallocate 75% of the blocks.
      const bool Deallocate = (std::rand() & 3) != 0;
      void *P = Allocator->allocate(
          Options, (static_cast<scudo::uptr>(std::rand()) % 16U) * PageSize);
      if (Deallocate)
        Allocator->deallocate(Options, P);
      else
        V.push_back(P);
    }
    while (!V.empty()) {
      Allocator->deallocate(Options, V.back());
      V.pop_back();
    }
  }

  std::mutex Mutex;
  std::condition_variable Cv;
  bool Ready = false;
};

TEST_F(MapAllocatorWithReleaseTest, SecondaryThreadsRace) {
  std::thread Threads[16];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] =
        std::thread(&MapAllocatorWithReleaseTest::performAllocations, this);
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  scudo::ScopedString Str;
  Allocator->getStats(&Str);
  Str.output();
}

struct MapAllocatorCacheTest : public Test {
  static constexpr scudo::u32 UnmappedMarker = 0xDEADBEEF;

  static void testUnmapCallback(scudo::MemMapT &MemMap) {
    scudo::u32 *Ptr = reinterpret_cast<scudo::u32 *>(MemMap.getBase());
    *Ptr = UnmappedMarker;
  }

  using SecondaryConfig = scudo::SecondaryConfig<TestConfig>;
  using CacheConfig = SecondaryConfig::CacheConfig;
  using CacheT = scudo::MapAllocatorCache<CacheConfig, testUnmapCallback>;

  std::unique_ptr<CacheT> Cache = std::make_unique<CacheT>();

  const scudo::uptr PageSize = scudo::getPageSizeCached();
  // The current test allocation size is set to the minimum size
  // needed for the scudo allocator to fall back to the secondary allocator
  static constexpr scudo::uptr TestAllocSize =
      CacheConfig::getDefaultMaxEntrySize();

  scudo::Options Options = getOptionsForConfig<SecondaryConfig>();

  void SetUp() override { Cache->init(/*ReleaseToOsInterval=*/-1); }

  void TearDown() override { Cache->unmapTestOnly(); }

  scudo::MemMapT allocate(scudo::uptr Size) {
    scudo::uptr MapSize = scudo::roundUp(Size, PageSize);
    scudo::ReservedMemoryT ReservedMemory;
    CHECK(ReservedMemory.create(0U, MapSize, nullptr, MAP_ALLOWNOMEM));

    scudo::MemMapT MemMap = ReservedMemory.dispatch(
        ReservedMemory.getBase(), ReservedMemory.getCapacity());
    MemMap.remap(MemMap.getBase(), MemMap.getCapacity(), "scudo:test",
                 MAP_RESIZABLE | MAP_ALLOWNOMEM);
    return MemMap;
  }

  void fillCacheWithSameSizeBlocks(std::vector<scudo::MemMapT> &MemMaps,
                                   scudo::uptr NumEntries, scudo::uptr Size) {
    for (scudo::uptr I = 0; I < NumEntries; I++) {
      MemMaps.emplace_back(allocate(Size));
      auto &MemMap = MemMaps[I];
      Cache->store(Options, MemMap.getBase(), MemMap.getCapacity(),
                   MemMap.getBase(), MemMap);
    }
  }
};

TEST_F(MapAllocatorCacheTest, CacheOrder) {
  std::vector<scudo::MemMapT> MemMaps;
  Cache->setOption(scudo::Option::MaxCacheEntriesCount,
                   CacheConfig::getEntriesArraySize());

  fillCacheWithSameSizeBlocks(MemMaps, CacheConfig::getEntriesArraySize(),
                              TestAllocSize);

  // Retrieval order should be the inverse of insertion order
  for (scudo::uptr I = CacheConfig::getEntriesArraySize(); I > 0; I--) {
    scudo::uptr EntryHeaderPos;
    scudo::CachedBlock Entry =
        Cache->retrieve(TestAllocSize, PageSize, 0, EntryHeaderPos);
    EXPECT_EQ(Entry.MemMap.getBase(), MemMaps[I - 1].getBase());
  }

  // Clean up MemMaps
  for (auto &MemMap : MemMaps)
    MemMap.unmap();
}

TEST_F(MapAllocatorCacheTest, MemoryLeakTest) {
  std::vector<scudo::MemMapT> MemMaps;
  // Fill the cache above MaxEntriesCount to force an eviction
  // The first cache entry should be evicted (because it is the oldest)
  // due to the maximum number of entries being reached
  fillCacheWithSameSizeBlocks(
      MemMaps, CacheConfig::getDefaultMaxEntriesCount() + 1, TestAllocSize);

  std::vector<scudo::CachedBlock> RetrievedEntries;

  // First MemMap should be evicted from cache because it was the first
  // inserted into the cache
  for (scudo::uptr I = CacheConfig::getDefaultMaxEntriesCount(); I > 0; I--) {
    scudo::uptr EntryHeaderPos;
    RetrievedEntries.push_back(
        Cache->retrieve(TestAllocSize, PageSize, 0, EntryHeaderPos));
    EXPECT_EQ(MemMaps[I].getBase(), RetrievedEntries.back().MemMap.getBase());
  }

  // Evicted entry should be marked due to unmap callback
  EXPECT_EQ(*reinterpret_cast<scudo::u32 *>(MemMaps[0].getBase()),
            UnmappedMarker);

  // Clean up MemMaps
  for (auto &MemMap : MemMaps)
    MemMap.unmap();
}
