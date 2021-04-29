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

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

// Note that with small enough regions, the SizeClassAllocator64 also works on
// 32-bit architectures. It's not something we want to encourage, but we still
// should ensure the tests pass.

struct TestConfig1 {
  static const scudo::uptr PrimaryRegionSizeLog = 18U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = false;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
};

struct TestConfig2 {
  static const scudo::uptr PrimaryRegionSizeLog = 24U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = false;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
};

struct TestConfig3 {
  static const scudo::uptr PrimaryRegionSizeLog = 24U;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  static const bool MaySupportMemoryTagging = true;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;
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
  SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TestConfig3)
#endif

#define SCUDO_TYPED_TEST_TYPE(FIXTURE, NAME, TYPE)                             \
  using FIXTURE##NAME##_##TYPE = FIXTURE##NAME<TYPE>;                          \
  TEST_F(FIXTURE##NAME##_##TYPE, NAME) { Run(); }

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
  scudo::ScopedString Str(1024);
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
  for (scudo::uptr I = 0; I < 10000U; I++) {
    TransferBatch *B = Allocator.popBatch(&Cache, ClassId);
    if (!B) {
      AllocationFailed = true;
      break;
    }
    for (scudo::u32 J = 0; J < B->getCount(); J++)
      memset(Allocator.decompactPtr(ClassId, B->get(J)), 'B', Size);
    Batches.push_back(B);
  }
  while (!Batches.empty()) {
    Allocator.pushBatch(ClassId, Batches.back());
    Batches.pop_back();
  }
  Cache.destroy(nullptr);
  Allocator.releaseToOS();
  scudo::ScopedString Str(1024);
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
  auto Lambda = [V, &Found](scudo::uptr Block) {
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
  scudo::ScopedString Str(1024);
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
  scudo::ScopedString Str(1024);
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
