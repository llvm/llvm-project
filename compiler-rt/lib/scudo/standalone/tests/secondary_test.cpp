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

#include <string.h>

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Get this once to use through-out the tests.
const scudo::uptr PageSize = scudo::getPageSizeCached();

template <typename Config> static scudo::Options getOptionsForConfig() {
  if (!Config::getMaySupportMemoryTagging() ||
      !scudo::archSupportsMemoryTagging() ||
      !scudo::systemSupportsMemoryTagging())
    return {};
  scudo::AtomicOptions AO;
  AO.set(scudo::OptionBit::UseMemoryTagging);
  return AO.load();
}

template <class Config> struct AllocatorInfoType {
  std::unique_ptr<scudo::MapAllocator<scudo::SecondaryConfig<Config>>>
      Allocator;
  scudo::GlobalStats GlobalStats;
  scudo::Options Options;

  AllocatorInfoType(scudo::s32 ReleaseToOsInterval) {
    using SecondaryT = scudo::MapAllocator<scudo::SecondaryConfig<Config>>;
    Options = getOptionsForConfig<scudo::SecondaryConfig<Config>>();
    GlobalStats.init();
    Allocator.reset(new SecondaryT);
    Allocator->init(&GlobalStats, ReleaseToOsInterval);
  }

  AllocatorInfoType() : AllocatorInfoType(-1) {}

  ~AllocatorInfoType() {
    if (Allocator == nullptr) {
      return;
    }

    if (TEST_HAS_FAILURE) {
      // Print all of the stats if the test fails.
      scudo::ScopedString Str;
      Allocator->getStats(&Str);
      Str.output();
    }

    Allocator->unmapTestOnly();
  }
};

struct TestNoCacheConfig {
  static const bool MaySupportMemoryTagging = false;
  template <typename> using TSDRegistryT = void;
  template <typename> using PrimaryT = void;
  template <typename Config> using SecondaryT = scudo::MapAllocator<Config>;

  struct Secondary {
    template <typename Config>
    using CacheT = scudo::MapAllocatorNoCache<Config>;
  };
};

struct TestCacheConfig {
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

template <typename Config> static void testBasic() {
  using SecondaryT = scudo::MapAllocator<scudo::SecondaryConfig<Config>>;
  AllocatorInfoType<Config> Info;

  const scudo::uptr Size = 1U << 16;
  void *P = Info.Allocator->allocate(Info.Options, Size);
  EXPECT_NE(P, nullptr);
  memset(P, 'A', Size);
  EXPECT_GE(SecondaryT::getBlockSize(P), Size);
  Info.Allocator->deallocate(Info.Options, P);

  // If the Secondary can't cache that pointer, it will be unmapped.
  if (!Info.Allocator->canCache(Size)) {
    EXPECT_DEATH(
        {
          // Repeat few time to avoid missing crash if it's mmaped by unrelated
          // code.
          for (int i = 0; i < 10; ++i) {
            P = Info.Allocator->allocate(Info.Options, Size);
            Info.Allocator->deallocate(Info.Options, P);
            memset(P, 'A', Size);
          }
        },
        "");
  }

  const scudo::uptr Align = 1U << 16;
  P = Info.Allocator->allocate(Info.Options, Size + Align, Align);
  EXPECT_NE(P, nullptr);
  void *AlignedP = reinterpret_cast<void *>(
      scudo::roundUp(reinterpret_cast<scudo::uptr>(P), Align));
  memset(AlignedP, 'A', Size);
  Info.Allocator->deallocate(Info.Options, P);

  std::vector<void *> V;
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(Info.Allocator->allocate(Info.Options, Size));
  std::shuffle(V.begin(), V.end(), std::mt19937(std::random_device()()));
  while (!V.empty()) {
    Info.Allocator->deallocate(Info.Options, V.back());
    V.pop_back();
  }
}

TEST(ScudoSecondaryTest, Basic) {
  testBasic<TestNoCacheConfig>();
  testBasic<TestCacheConfig>();
  testBasic<scudo::DefaultConfig>();
}

// This exercises a variety of combinations of size and alignment for the
// MapAllocator. The size computation done here mimic the ones done by the
// combined allocator.
TEST(ScudoSecondaryTest, AllocatorCombinations) {
  AllocatorInfoType<TestNoCacheConfig> Info;

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
        void *P = Info.Allocator->allocate(Info.Options, Size, Align);
        EXPECT_NE(P, nullptr);
        void *AlignedP = reinterpret_cast<void *>(
            scudo::roundUp(reinterpret_cast<scudo::uptr>(P), Align));
        memset(AlignedP, 0xff, UserSize);
        Info.Allocator->deallocate(Info.Options, P);
      }
    }
  }
}

TEST(ScudoSecondaryTest, AllocatorIterate) {
  AllocatorInfoType<TestNoCacheConfig> Info;

  std::vector<void *> V;
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(Info.Allocator->allocate(
        Info.Options,
        (static_cast<scudo::uptr>(std::rand()) % 16U) * PageSize));
  auto Lambda = [&V](scudo::uptr Block) {
    EXPECT_NE(std::find(V.begin(), V.end(), reinterpret_cast<void *>(Block)),
              V.end());
  };
  Info.Allocator->disable();
  Info.Allocator->iterateOverBlocks(Lambda);
  Info.Allocator->enable();
  while (!V.empty()) {
    Info.Allocator->deallocate(Info.Options, V.back());
    V.pop_back();
  }
}

TEST(ScudoSecondaryTest, AllocatorWithReleaseThreadsRace) {
  AllocatorInfoType<TestNoCacheConfig> Info(/*ReleaseToOsInterval=*/0);

  std::mutex Mutex;
  std::condition_variable Cv;
  bool Ready = false;

  std::thread Threads[16];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread([&Mutex, &Cv, &Ready, &Info]() {
      std::vector<void *> V;
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        while (!Ready)
          Cv.wait(Lock);
      }
      for (scudo::uptr I = 0; I < 128U; I++) {
        // Deallocate 75% of the blocks.
        const bool Deallocate = (std::rand() & 3) != 0;
        void *P = Info.Allocator->allocate(
            Info.Options,
            (static_cast<scudo::uptr>(std::rand()) % 16U) * PageSize);
        if (Deallocate)
          Info.Allocator->deallocate(Info.Options, P);
        else
          V.push_back(P);
      }
      while (!V.empty()) {
        Info.Allocator->deallocate(Info.Options, V.back());
        V.pop_back();
      }
    });

  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
}

// Value written to cache entries that are unmapped.
static scudo::u32 UnmappedMarker = 0xDEADBEEF;

template <class Config> struct CacheInfoType {
  static void addMarkerToMapCallback(scudo::MemMapT &MemMap) {
    // When a cache entry is unmaped, don't unmap it write a special marker
    // to indicate the cache entry was released. The real unmap will happen
    // in the destructor. It is assumed that all of these maps will be in
    // the MemMaps vector.
    scudo::u32 *Ptr = reinterpret_cast<scudo::u32 *>(MemMap.getBase());
    *Ptr = UnmappedMarker;
  }

  using SecondaryConfig = scudo::SecondaryConfig<TestCacheConfig>;
  using CacheConfig = SecondaryConfig::CacheConfig;
  using CacheT = scudo::MapAllocatorCache<CacheConfig, addMarkerToMapCallback>;
  scudo::Options Options = getOptionsForConfig<SecondaryConfig>();
  std::unique_ptr<CacheT> Cache = std::make_unique<CacheT>();
  std::vector<scudo::MemMapT> MemMaps;
  // The current test allocation size is set to the maximum
  // cache entry size
  static constexpr scudo::uptr TestAllocSize =
      CacheConfig::getDefaultMaxEntrySize();

  CacheInfoType() { Cache->init(/*ReleaseToOsInterval=*/-1); }

  ~CacheInfoType() {
    if (Cache == nullptr) {
      return;
    }

    // Clean up MemMaps
    for (auto &MemMap : MemMaps)
      MemMap.unmap();
  }

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

  void fillCacheWithSameSizeBlocks(scudo::uptr NumEntries, scudo::uptr Size) {
    for (scudo::uptr I = 0; I < NumEntries; I++) {
      MemMaps.emplace_back(allocate(Size));
      auto &MemMap = MemMaps[I];
      Cache->store(Options, MemMap.getBase(), MemMap.getCapacity(),
                   MemMap.getBase(), MemMap);
    }
  }
};

TEST(ScudoSecondaryTest, AllocatorCacheEntryOrder) {
  CacheInfoType<TestCacheConfig> Info;
  using CacheConfig = CacheInfoType<TestCacheConfig>::CacheConfig;

  Info.Cache->setOption(scudo::Option::MaxCacheEntriesCount,
                        CacheConfig::getEntriesArraySize());

  Info.fillCacheWithSameSizeBlocks(CacheConfig::getEntriesArraySize(),
                                   Info.TestAllocSize);

  // Retrieval order should be the inverse of insertion order
  for (scudo::uptr I = CacheConfig::getEntriesArraySize(); I > 0; I--) {
    scudo::uptr EntryHeaderPos;
    scudo::CachedBlock Entry = Info.Cache->retrieve(
        0, Info.TestAllocSize, PageSize, 0, EntryHeaderPos);
    EXPECT_EQ(Entry.MemMap.getBase(), Info.MemMaps[I - 1].getBase());
  }
}

TEST(ScudoSecondaryTest, AllocatorCachePartialChunkHeuristicRetrievalTest) {
  CacheInfoType<TestCacheConfig> Info;

  const scudo::uptr FragmentedPages =
      1 + scudo::CachedBlock::MaxReleasedCachePages;
  scudo::uptr EntryHeaderPos;
  scudo::CachedBlock Entry;
  scudo::MemMapT MemMap = Info.allocate(PageSize + FragmentedPages * PageSize);
  Info.Cache->store(Info.Options, MemMap.getBase(), MemMap.getCapacity(),
                    MemMap.getBase(), MemMap);

  // FragmentedPages > MaxAllowedFragmentedPages so PageSize
  // cannot be retrieved from the cache
  Entry = Info.Cache->retrieve(/*MaxAllowedFragmentedPages=*/0, PageSize,
                               PageSize, 0, EntryHeaderPos);
  EXPECT_FALSE(Entry.isValid());

  // FragmentedPages == MaxAllowedFragmentedPages so PageSize
  // can be retrieved from the cache
  Entry = Info.Cache->retrieve(FragmentedPages, PageSize, PageSize, 0,
                               EntryHeaderPos);
  EXPECT_TRUE(Entry.isValid());

  MemMap.unmap();
}

TEST(ScudoSecondaryTest, AllocatorCacheMemoryLeakTest) {
  CacheInfoType<TestCacheConfig> Info;
  using CacheConfig = CacheInfoType<TestCacheConfig>::CacheConfig;

  // Fill the cache above MaxEntriesCount to force an eviction
  // The first cache entry should be evicted (because it is the oldest)
  // due to the maximum number of entries being reached
  Info.fillCacheWithSameSizeBlocks(CacheConfig::getDefaultMaxEntriesCount() + 1,
                                   Info.TestAllocSize);

  std::vector<scudo::CachedBlock> RetrievedEntries;

  // First MemMap should be evicted from cache because it was the first
  // inserted into the cache
  for (scudo::uptr I = CacheConfig::getDefaultMaxEntriesCount(); I > 0; I--) {
    scudo::uptr EntryHeaderPos;
    RetrievedEntries.push_back(Info.Cache->retrieve(
        0, Info.TestAllocSize, PageSize, 0, EntryHeaderPos));
    EXPECT_EQ(Info.MemMaps[I].getBase(),
              RetrievedEntries.back().MemMap.getBase());
  }

  // Evicted entry should be marked due to unmap callback
  EXPECT_EQ(*reinterpret_cast<scudo::u32 *>(Info.MemMaps[0].getBase()),
            UnmappedMarker);
}

TEST(ScudoSecondaryTest, AllocatorCacheOptions) {
  CacheInfoType<TestCacheConfig> Info;

  // Attempt to set a maximum number of entries higher than the array size.
  EXPECT_TRUE(
      Info.Cache->setOption(scudo::Option::MaxCacheEntriesCount, 4096U));

  // Attempt to set an invalid (negative) number of entries
  EXPECT_FALSE(Info.Cache->setOption(scudo::Option::MaxCacheEntriesCount, -1));

  // Various valid combinations.
  EXPECT_TRUE(Info.Cache->setOption(scudo::Option::MaxCacheEntriesCount, 4U));
  EXPECT_TRUE(
      Info.Cache->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 20));
  EXPECT_TRUE(Info.Cache->canCache(1UL << 18));
  EXPECT_TRUE(
      Info.Cache->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 17));
  EXPECT_FALSE(Info.Cache->canCache(1UL << 18));
  EXPECT_TRUE(Info.Cache->canCache(1UL << 16));
  EXPECT_TRUE(Info.Cache->setOption(scudo::Option::MaxCacheEntriesCount, 0U));
  EXPECT_FALSE(Info.Cache->canCache(1UL << 16));
  EXPECT_TRUE(Info.Cache->setOption(scudo::Option::MaxCacheEntriesCount, 4U));
  EXPECT_TRUE(
      Info.Cache->setOption(scudo::Option::MaxCacheEntrySize, 1UL << 20));
  EXPECT_TRUE(Info.Cache->canCache(1UL << 16));
}
