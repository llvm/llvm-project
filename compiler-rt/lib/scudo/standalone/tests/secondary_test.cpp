//===-- secondary_test.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "secondary.h"

#include "gtest/gtest.h"

#include <stdio.h>

#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

template <class SecondaryT> static void testSecondaryBasic(void) {
  scudo::GlobalStats S;
  S.init();
  SecondaryT *L = new SecondaryT;
  L->init(&S);
  const scudo::uptr Size = 1U << 16;
  void *P = L->allocate(Size);
  EXPECT_NE(P, nullptr);
  memset(P, 'A', Size);
  EXPECT_GE(SecondaryT::getBlockSize(P), Size);
  L->deallocate(P);
  // If we are not using a free list, blocks are unmapped on deallocation.
  if (SecondaryT::getMaxFreeListSize() == 0U)
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

TEST(ScudoSecondaryTest, SecondaryBasic) {
  testSecondaryBasic<scudo::MapAllocator<>>();
  testSecondaryBasic<scudo::MapAllocator<0U>>();
  testSecondaryBasic<scudo::MapAllocator<64U>>();
}

using LargeAllocator = scudo::MapAllocator<>;

// This exercises a variety of combinations of size and alignment for the
// MapAllocator. The size computation done here mimic the ones done by the
// combined allocator.
TEST(ScudoSecondaryTest, SecondaryCombinations) {
  constexpr scudo::uptr MinAlign = FIRST_32_SECOND_64(8, 16);
  constexpr scudo::uptr HeaderSize = scudo::roundUpTo(8, MinAlign);
  LargeAllocator *L = new LargeAllocator;
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
  LargeAllocator *L = new LargeAllocator;
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

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready = false;

static void performAllocations(LargeAllocator *L) {
  std::vector<void *> V;
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  for (scudo::uptr I = 0; I < 32U; I++)
    V.push_back(L->allocate((std::rand() % 16) * PageSize));
  while (!V.empty()) {
    L->deallocate(V.back());
    V.pop_back();
  }
}

TEST(ScudoSecondaryTest, SecondaryThreadsRace) {
  LargeAllocator *L = new LargeAllocator;
  L->init(nullptr);
  std::thread Threads[10];
  for (scudo::uptr I = 0; I < 10U; I++)
    Threads[I] = std::thread(performAllocations, L);
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
