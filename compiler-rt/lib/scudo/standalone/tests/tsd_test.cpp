//===-- tsd_test.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tsd_exclusive.h"
#include "tsd_shared.h"

#include "gtest/gtest.h"

#include <condition_variable>
#include <mutex>
#include <thread>

// We mock out an allocator with a TSD registry, mostly using empty stubs. The
// cache contains a single volatile uptr, to be able to test that several
// concurrent threads will not access or modify the same cache at the same time.
template <class Config> class MockAllocator {
public:
  using ThisT = MockAllocator<Config>;
  using TSDRegistryT = typename Config::template TSDRegistryT<ThisT>;
  using CacheT = struct MockCache { volatile scudo::uptr Canary; };
  using QuarantineCacheT = struct MockQuarantine {};

  void initLinkerInitialized() {
    // This should only be called once by the registry.
    EXPECT_FALSE(Initialized);
    Initialized = true;
  }
  void reset() { memset(this, 0, sizeof(*this)); }

  void unmapTestOnly() { TSDRegistry.unmapTestOnly(); }
  void initCache(CacheT *Cache) { memset(Cache, 0, sizeof(*Cache)); }
  void commitBack(scudo::TSD<MockAllocator> *TSD) {}
  TSDRegistryT *getTSDRegistry() { return &TSDRegistry; }

  bool isInitialized() { return Initialized; }

private:
  bool Initialized;
  TSDRegistryT TSDRegistry;
};

struct OneCache {
  template <class Allocator>
  using TSDRegistryT = scudo::TSDRegistrySharedT<Allocator, 1U>;
};

struct SharedCaches {
  template <class Allocator>
  using TSDRegistryT = scudo::TSDRegistrySharedT<Allocator, 16U>;
};

struct ExclusiveCaches {
  template <class Allocator>
  using TSDRegistryT = scudo::TSDRegistryExT<Allocator>;
};

TEST(ScudoTSDTest, TSDRegistryInit) {
  using AllocatorT = MockAllocator<OneCache>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();
  EXPECT_FALSE(Allocator->isInitialized());

  auto Registry = Allocator->getTSDRegistry();
  Registry->initLinkerInitialized(Allocator.get());
  EXPECT_TRUE(Allocator->isInitialized());
}

template <class AllocatorT> static void testRegistry() {
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();
  EXPECT_FALSE(Allocator->isInitialized());

  auto Registry = Allocator->getTSDRegistry();
  Registry->initThreadMaybe(Allocator.get(), /*MinimalInit=*/true);
  EXPECT_TRUE(Allocator->isInitialized());

  bool UnlockRequired;
  auto TSD = Registry->getTSDAndLock(&UnlockRequired);
  EXPECT_NE(TSD, nullptr);
  EXPECT_EQ(TSD->Cache.Canary, 0U);
  if (UnlockRequired)
    TSD->unlock();

  Registry->initThreadMaybe(Allocator.get(), /*MinimalInit=*/false);
  TSD = Registry->getTSDAndLock(&UnlockRequired);
  EXPECT_NE(TSD, nullptr);
  EXPECT_EQ(TSD->Cache.Canary, 0U);
  memset(&TSD->Cache, 0x42, sizeof(TSD->Cache));
  if (UnlockRequired)
    TSD->unlock();
}

TEST(ScudoTSDTest, TSDRegistryBasic) {
  testRegistry<MockAllocator<OneCache>>();
  testRegistry<MockAllocator<SharedCaches>>();
  testRegistry<MockAllocator<ExclusiveCaches>>();
}

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready = false;

template <typename AllocatorT> static void stressCache(AllocatorT *Allocator) {
  auto Registry = Allocator->getTSDRegistry();
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  Registry->initThreadMaybe(Allocator, /*MinimalInit=*/false);
  bool UnlockRequired;
  auto TSD = Registry->getTSDAndLock(&UnlockRequired);
  EXPECT_NE(TSD, nullptr);
  // For an exclusive TSD, the cache should be empty. We cannot guarantee the
  // same for a shared TSD.
  if (!UnlockRequired)
    EXPECT_EQ(TSD->Cache.Canary, 0U);
  // Transform the thread id to a uptr to use it as canary.
  const scudo::uptr Canary = static_cast<scudo::uptr>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  TSD->Cache.Canary = Canary;
  // Loop a few times to make sure that a concurrent thread isn't modifying it.
  for (scudo::uptr I = 0; I < 4096U; I++)
    EXPECT_EQ(TSD->Cache.Canary, Canary);
  if (UnlockRequired)
    TSD->unlock();
}

template <class AllocatorT> static void testRegistryThreaded() {
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();
  std::thread Threads[32];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(stressCache<AllocatorT>, Allocator.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
}

TEST(ScudoTSDTest, TSDRegistryThreaded) {
  testRegistryThreaded<MockAllocator<OneCache>>();
  testRegistryThreaded<MockAllocator<SharedCaches>>();
  testRegistryThreaded<MockAllocator<ExclusiveCaches>>();
}
