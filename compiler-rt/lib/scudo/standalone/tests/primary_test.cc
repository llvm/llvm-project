//===-- primary_test.cc -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "primary32.h"
#include "primary64.h"
#include "size_class_map.h"

#include "gtest/gtest.h"

#include <condition_variable>
#include <mutex>
#include <thread>

// Note that with small enough regions, the SizeClassAllocator64 also works on
// 32-bit architectures. It's not something we want to encourage, but we still
// should ensure the tests pass.

template <typename Primary> static void testPrimary() {
  const scudo::uptr NumberOfAllocations = 32U;
  auto Deleter = [](Primary *P) {
    P->unmapTestOnly();
    delete P;
  };
  std::unique_ptr<Primary, decltype(Deleter)> Allocator(new Primary, Deleter);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  typename Primary::CacheT Cache;
  Cache.init(nullptr, Allocator.get());
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
  Allocator->printStats();
}

TEST(ScudoPrimaryTest, BasicPrimary) {
  using SizeClassMap = scudo::DefaultSizeClassMap;
  testPrimary<scudo::SizeClassAllocator32<SizeClassMap, 18U>>();
  testPrimary<scudo::SizeClassAllocator64<SizeClassMap, 24U>>();
}

// The 64-bit SizeClassAllocator can be easily OOM'd with small region sizes.
// For the 32-bit one, it requires actually exhausting memory, so we skip it.
TEST(ScudoPrimaryTest, Primary64OOM) {
  using Primary = scudo::SizeClassAllocator64<scudo::DefaultSizeClassMap, 20U>;
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
    for (scudo::uptr J = 0; J < B->getCount(); J++)
      memset(B->get(J), 'B', Size);
    Batches.push_back(B);
  }
  while (!Batches.empty()) {
    Allocator.pushBatch(ClassId, Batches.back());
    Batches.pop_back();
  }
  Cache.destroy(nullptr);
  Allocator.releaseToOS();
  Allocator.printStats();
  EXPECT_EQ(AllocationFailed, true);
  Allocator.unmapTestOnly();
}

template <typename Primary> static void testIteratePrimary() {
  auto Deleter = [](Primary *P) {
    P->unmapTestOnly();
    delete P;
  };
  std::unique_ptr<Primary, decltype(Deleter)> Allocator(new Primary, Deleter);
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
  Allocator->printStats();
}

TEST(ScudoPrimaryTest, PrimaryIterate) {
  using SizeClassMap = scudo::DefaultSizeClassMap;
  testIteratePrimary<scudo::SizeClassAllocator32<SizeClassMap, 18U>>();
  testIteratePrimary<scudo::SizeClassAllocator64<SizeClassMap, 24U>>();
}

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready = false;

template <typename Primary> static void performAllocations(Primary *Allocator) {
  static THREADLOCAL typename Primary::CacheT Cache;
  Cache.init(nullptr, Allocator);
  std::vector<std::pair<scudo::uptr, void *>> V;
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  for (scudo::uptr I = 0; I < 256U; I++) {
    const scudo::uptr Size = std::rand() % Primary::SizeClassMap::MaxSize / 4;
    const scudo::uptr ClassId = Primary::SizeClassMap::getClassIdBySize(Size);
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
}

template <typename Primary> static void testPrimaryThreaded() {
  auto Deleter = [](Primary *P) {
    P->unmapTestOnly();
    delete P;
  };
  std::unique_ptr<Primary, decltype(Deleter)> Allocator(new Primary, Deleter);
  Allocator->init(/*ReleaseToOsInterval=*/-1);
  std::thread Threads[32];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(performAllocations<Primary>, Allocator.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  Allocator->releaseToOS();
  Allocator->printStats();
}

TEST(ScudoPrimaryTest, PrimaryThreaded) {
  using SizeClassMap = scudo::SvelteSizeClassMap;
  testPrimaryThreaded<scudo::SizeClassAllocator32<SizeClassMap, 18U>>();
  testPrimaryThreaded<scudo::SizeClassAllocator64<SizeClassMap, 24U>>();
}
