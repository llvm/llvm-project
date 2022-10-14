//===-- primary_test.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "primary32.h"
#include "primary64.h"
#include "size_class_map.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <random>
#include <stdlib.h>
#include <thread>
#include <vector>

// Note that with small enough regions, the SizeClassAllocator64 also works on
// 32-bit architectures. It's not something we want to encourage, but we still
// should ensure the tests pass.

struct TestConfig1 {
  static const scudo::uptr PrimaryRegionSizeLog = 18U;
  static const scudo::uptr PrimaryGroupSizeLog = 18U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = false;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
  static const bool PrimaryEnableRandomOffset = true;
  static const scudo::uptr PrimaryMapSizeIncrement = 1UL << 18;
};

struct TestConfig2 {
#if defined(__mips__)
  // Unable to allocate greater size on QEMU-user.
  static const scudo::uptr PrimaryRegionSizeLog = 23U;
#else
  static const scudo::uptr PrimaryRegionSizeLog = 24U;
#endif
  static const scudo::uptr PrimaryGroupSizeLog = 20U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = false;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
  static const bool PrimaryEnableRandomOffset = true;
  static const scudo::uptr PrimaryMapSizeIncrement = 1UL << 18;
};

struct TestConfig3 {
#if defined(__mips__)
  // Unable to allocate greater size on QEMU-user.
  static const scudo::uptr PrimaryRegionSizeLog = 23U;
#else
  static const scudo::uptr PrimaryRegionSizeLog = 24U;
#endif
  static const scudo::uptr PrimaryGroupSizeLog = 20U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = true;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
  static const bool PrimaryEnableRandomOffset = true;
  static const scudo::uptr PrimaryMapSizeIncrement = 1UL << 18;
};

struct TestConfig4 {
#if defined(__mips__)
  // Unable to allocate greater size on QEMU-user.
  static const scudo::uptr PrimaryRegionSizeLog = 23U;
#else
  static const scudo::uptr PrimaryRegionSizeLog = 24U;
#endif
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = true;
  static const scudo::uptr PrimaryCompactPtrScale = 3U;
  static const scudo::uptr PrimaryGroupSizeLog = 20U;
  typedef scudo::u32 PrimaryCompactPtrT;
  static const bool PrimaryEnableRandomOffset = true;
  static const scudo::uptr PrimaryMapSizeIncrement = 1UL << 18;
};

template <typename BaseConfig, typename SizeClassMapT>
struct Config : public BaseConfig {
  using SizeClassMap = SizeClassMapT;
};

template <typename BaseConfig, typename SizeClassMapT>
struct SizeClassAllocator
    : public scudo::SizeClassAllocator64<Config<BaseConfig, SizeClassMapT>> {};
template <typename SizeClassMapT>
struct SizeClassAllocator<TestConfig1, SizeClassMapT>
    : public scudo::SizeClassAllocator32<Config<TestConfig1, SizeClassMapT>> {};

template <typename BaseConfig, typename SizeClassMapT>
struct TestAllocator : public SizeClassAllocator<BaseConfig, SizeClassMapT> {
  ~TestAllocator() { this->unmapTestOnly(); }

  void *operator new(size_t size) {
    void *p = nullptr;
    EXPECT_EQ(0, posix_memalign(&p, alignof(TestAllocator), size));
    return p;
  }

  void operator delete(void *ptr) { free(ptr); }
};

template <class BaseConfig> struct ScudoPrimaryTest : public Test {};

#if SCUDO_FUCHSIA
#define SCUDO_TYPED_TEST_ALL_TYPES(FIXTURE, NAME)                              \
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig2)                            \
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig3)
#else
#define SCUDO_TYPED_TEST_ALL_TYPES(FIXTURE, NAME)                              \
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig1)                            \
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig2)                            \
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig3)                            \
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig4)
#endif

#define SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TYPE)                             \
  using FIXTURE##NAME##_##TYPE = FIXTURE##NAME<TYPE>;                          \
  TEST_F(FIXTURE##NAME##_##TYPE, NAME) { FIXTURE##NAME<TYPE>::Run(); }

#define SCUDO_TYPED_TEST(FIXTURE, NAME)                                        \
  template <class TypeParam>                                                   \
  struct FIXTURE##NAME : public FIXTURE<TypeParam> {                           \
    void Run();                                                                \
  };                                                                           \
  SCUDO_TYPED_TEST_ALL_TYPES(FIXTURE, NAME)                                    \
  template <class TypeParam> void FIXTURE##NAME<TypeParam>::Run()

SCUDO_TYPED_TEST(ScudoPrimaryTest, BasicPrimary) {
  using Primary = TestAllocator<TypeParam, scudo::DefaultSizeClassMap>;
  std::unique_ptr<Primary> Allocator(new Primary);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  typename Primary::CacheT Cache;
  Cache.init(nullptr, Allocator.get());
  const scudo::uptr NumberOfAllocations = 32U;
  for (scudo::uptr I = 0; I <= 16U; I++) {
    const scudo::uptr Size = 1UL << I;
    if (!Primary::canAllocate(Size))
      continue;
    const scudo::uptr ClassId = Primary::SizeClassMap::getClassIdBySize(Size);
    void *Pointers[NumberOfAllocations];
    for (scudo::uptr J = 0; J < NumberOfAllocations; J++) {
      void *P = Cache.allocate(ClassId);
      memset(P, 'B', Size);
      Pointers[J] = P;
    }
    for (scudo::uptr J = 0; J < NumberOfAllocations; J++)
      Cache.deallocate(ClassId, Pointers[J]);
  }
  Cache.destroy(nullptr);
  Allocator->releaseToOS();
  scudo::ScopedString Str;
  Allocator->getStats(&Str);
  Str.output();
}

struct SmallRegionsConfig {
  using SizeClassMap = scudo::DefaultSizeClassMap;
  static const scudo::uptr PrimaryRegionSizeLog = 20U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = false;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
  static const bool PrimaryEnableRandomOffset = true;
  static const scudo::uptr PrimaryMapSizeIncrement = 1UL << 18;
  static const scudo::uptr PrimaryGroupSizeLog = 20U;
};

// The 64-bit SizeClassAllocator can be easily OOM'd with small region sizes.
// For the 32-bit one, it requires actually exhausting memory, so we skip it.
TEST(ScudoPrimaryTest, Primary64OOM) {
  using Primary = scudo::SizeClassAllocator64<SmallRegionsConfig>;
  using TransferBatch = Primary::CacheT::TransferBatch;
  Primary Allocator;
  Allocator.init(/*ReleaseToOsInterval=*/-1);
  typename Primary::CacheT Cache;
  scudo::GlobalStats Stats;
  Stats.init();
  Cache.init(&Stats, &Allocator);
  bool AllocationFailed = false;
  std::vector<TransferBatch *> Batches;
  const scudo::uptr ClassId = Primary::SizeClassMap::LargestClassId;
  const scudo::uptr Size = Primary::getSizeByClassId(ClassId);
  typename Primary::CacheT::CompactPtrT Blocks[TransferBatch::MaxNumCached];

  for (scudo::uptr I = 0; I < 10000U; I++) {
    TransferBatch *B = Allocator.popBatch(&Cache, ClassId);
    if (!B) {
      AllocationFailed = true;
      break;
    }
    for (scudo::u16 J = 0; J < B->getCount(); J++)
      memset(Allocator.decompactPtr(ClassId, B->get(J)), 'B', Size);
    Batches.push_back(B);
  }
  while (!Batches.empty()) {
    TransferBatch *B = Batches.back();
    Batches.pop_back();
    B->copyToArray(Blocks);
    Allocator.pushBlocks(&Cache, ClassId, Blocks, B->getCount());
    Cache.deallocate(Primary::SizeClassMap::BatchClassId, B);
  }
  Cache.destroy(nullptr);
  Allocator.releaseToOS();
  scudo::ScopedString Str;
  Allocator.getStats(&Str);
  Str.output();
  EXPECT_EQ(AllocationFailed, true);
  Allocator.unmapTestOnly();
}

SCUDO_TYPED_TEST(ScudoPrimaryTest, PrimaryIterate) {
  using Primary = TestAllocator<TypeParam, scudo::DefaultSizeClassMap>;
  std::unique_ptr<Primary> Allocator(new Primary);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  typename Primary::CacheT Cache;
  Cache.init(nullptr, Allocator.get());
  std::vector<std::pair<scudo::uptr, void *>> V;
  for (scudo::uptr I = 0; I < 64U; I++) {
    const scudo::uptr Size = std::rand() % Primary::SizeClassMap::MaxSize;
    const scudo::uptr ClassId = Primary::SizeClassMap::getClassIdBySize(Size);
    void *P = Cache.allocate(ClassId);
    V.push_back(std::make_pair(ClassId, P));
  }
  scudo::uptr Found = 0;
  auto Lambda = [&V, &Found](scudo::uptr Block) {
    for (const auto &Pair : V) {
      if (Pair.second == reinterpret_cast<void *>(Block))
        Found++;
    }
  };
  Allocator->disable();
  Allocator->iterateOverBlocks(Lambda);
  Allocator->enable();
  EXPECT_EQ(Found, V.size());
  while (!V.empty()) {
    auto Pair = V.back();
    Cache.deallocate(Pair.first, Pair.second);
    V.pop_back();
  }
  Cache.destroy(nullptr);
  Allocator->releaseToOS();
  scudo::ScopedString Str;
  Allocator->getStats(&Str);
  Str.output();
}

SCUDO_TYPED_TEST(ScudoPrimaryTest, PrimaryThreaded) {
  using Primary = TestAllocator<TypeParam, scudo::SvelteSizeClassMap>;
  std::unique_ptr<Primary> Allocator(new Primary);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  std::mutex Mutex;
  std::condition_variable Cv;
  bool Ready = false;
  std::thread Threads[32];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread([&]() {
      static thread_local typename Primary::CacheT Cache;
      Cache.init(nullptr, Allocator.get());
      std::vector<std::pair<scudo::uptr, void *>> V;
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        while (!Ready)
          Cv.wait(Lock);
      }
      for (scudo::uptr I = 0; I < 256U; I++) {
        const scudo::uptr Size =
            std::rand() % Primary::SizeClassMap::MaxSize / 4;
        const scudo::uptr ClassId =
            Primary::SizeClassMap::getClassIdBySize(Size);
        void *P = Cache.allocate(ClassId);
        if (P)
          V.push_back(std::make_pair(ClassId, P));
      }
      while (!V.empty()) {
        auto Pair = V.back();
        Cache.deallocate(Pair.first, Pair.second);
        V.pop_back();
      }
      Cache.destroy(nullptr);
    });
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  Allocator->releaseToOS();
  scudo::ScopedString Str;
  Allocator->getStats(&Str);
  Str.output();
}

// Through a simple allocation that spans two pages, verify that releaseToOS
// actually releases some bytes (at least one page worth). This is a regression
// test for an error in how the release criteria were computed.
SCUDO_TYPED_TEST(ScudoPrimaryTest, ReleaseToOS) {
  using Primary = TestAllocator<TypeParam, scudo::DefaultSizeClassMap>;
  std::unique_ptr<Primary> Allocator(new Primary);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  typename Primary::CacheT Cache;
  Cache.init(nullptr, Allocator.get());
  const scudo::uptr Size = scudo::getPageSizeCached() * 2;
  EXPECT_TRUE(Primary::canAllocate(Size));
  const scudo::uptr ClassId = Primary::SizeClassMap::getClassIdBySize(Size);
  void *P = Cache.allocate(ClassId);
  EXPECT_NE(P, nullptr);
  Cache.deallocate(ClassId, P);
  Cache.destroy(nullptr);
  EXPECT_GT(Allocator->releaseToOS(), 0U);
}

SCUDO_TYPED_TEST(ScudoPrimaryTest, MemoryGroup) {
  using Primary = TestAllocator<TypeParam, scudo::DefaultSizeClassMap>;
  std::unique_ptr<Primary> Allocator(new Primary);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  typename Primary::CacheT Cache;
  Cache.init(nullptr, Allocator.get());
  const scudo::uptr Size = 32U;
  const scudo::uptr ClassId = Primary::SizeClassMap::getClassIdBySize(Size);

  // We will allocate 4 times the group size memory and release all of them. We
  // expect the free blocks will be classified with groups. Then we will
  // allocate the same amount of memory as group size and expect the blocks will
  // have the max address difference smaller or equal to 2 times the group size.
  // Note that it isn't necessary to be in the range of single group size
  // because the way we get the group id is doing compact pointer shifting.
  // According to configuration, the compact pointer may not align to group
  // size. As a result, the blocks can cross two groups at most.
  const scudo::uptr GroupSizeMem = (1ULL << Primary::GroupSizeLog);
  const scudo::uptr PeakAllocationMem = 4 * GroupSizeMem;
  const scudo::uptr PeakNumberOfAllocations = PeakAllocationMem / Size;
  const scudo::uptr FinalNumberOfAllocations = GroupSizeMem / Size;
  std::vector<scudo::uptr> Blocks;
  std::mt19937 R;

  for (scudo::uptr I = 0; I < PeakNumberOfAllocations; ++I)
    Blocks.push_back(reinterpret_cast<scudo::uptr>(Cache.allocate(ClassId)));

  std::shuffle(Blocks.begin(), Blocks.end(), R);

  // Release all the allocated blocks, including those held by local cache.
  while (!Blocks.empty()) {
    Cache.deallocate(ClassId, reinterpret_cast<void *>(Blocks.back()));
    Blocks.pop_back();
  }
  Cache.drain();

  for (scudo::uptr I = 0; I < FinalNumberOfAllocations; ++I)
    Blocks.push_back(reinterpret_cast<scudo::uptr>(Cache.allocate(ClassId)));

  EXPECT_LE(*std::max_element(Blocks.begin(), Blocks.end()) -
            *std::min_element(Blocks.begin(), Blocks.end()), GroupSizeMem * 2);
}
