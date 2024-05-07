//===--- radsan_test.cpp - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "radsan_test_utilities.h"
#include <radsan.h>
#include <sanitizer_common/sanitizer_platform.h>
#include <sanitizer_common/sanitizer_platform_interceptors.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <thread>

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&                  \
    __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 101200
#define SI_MAC_DEPLOYMENT_AT_LEAST_10_12 1
#else
#define SI_MAC_DEPLOYMENT_AT_LEAST_10_12 0
#endif

#define RADSAN_TEST_SHARED_MUTEX (!(SI_MAC) || SI_MAC_DEPLOYMENT_AT_LEAST_10_12)

using namespace testing;
using namespace radsan_testing;
using namespace std::chrono_literals;

TEST(TestRadsan, VectorPushBackAllocationDiesWhenRealtime) {
  std::vector<float> Vec{};
  auto Func = [&Vec]() { Vec.push_back(0.4f); };
  ExpectRealtimeDeath(Func);
  ASSERT_EQ(0u, Vec.size());
  ExpectNonRealtimeSurvival(Func);
  ASSERT_EQ(1u, Vec.size());
}

TEST(TestRadsan, DestructionOfObjectOnHeapDiesWhenRealtime) {
  auto AllocatedPtr = std::make_unique<std::array<float, 256>>();
  auto Func = [&AllocatedPtr]() { AllocatedPtr.reset(); };
  ExpectRealtimeDeath(Func);
  ASSERT_NE(nullptr, AllocatedPtr.get());
  ExpectNonRealtimeSurvival(Func);
  ASSERT_EQ(nullptr, AllocatedPtr.get());
}

TEST(TestRadsan, SleepingAThreadDiesWhenRealtime) {
  auto Func = []() { std::this_thread::sleep_for(1us); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, IfstreamCreationDiesWhenRealtime) {
  auto Func = []() { std::ifstream ifs{"./file.txt"}; };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
  std::remove("./file.txt");
}

TEST(TestRadsan, OfstreamCreationDiesWhenRealtime) {
  auto Func = []() { std::ofstream ofs{"./file.txt"}; };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
  std::remove("./file.txt");
}

TEST(TestRadsan, LockingAMutexDiesWhenRealtime) {
  std::mutex Mutex{};
  auto Func = [&]() { Mutex.lock(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, UnlockingAMutexDiesWhenRealtime) {
  std::mutex Mutex{};
  Mutex.lock();
  auto Func = [&]() { Mutex.unlock(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

#if RADSAN_TEST_SHARED_MUTEX

TEST(TestRadsan, LockingASharedMutexDiesWhenRealtime) {
  std::shared_mutex Mutex{};
  auto Func = [&]() { Mutex.lock(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, UnlockingASharedMutexDiesWhenRealtime) {
  std::shared_mutex Mutex{};
  Mutex.lock();
  auto Func = [&]() { Mutex.unlock(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, SharedLockingASharedMutexDiesWhenRealtime) {
  std::shared_mutex Mutex{};
  auto Func = [&]() { Mutex.lock_shared(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, SharedUnlockingASharedMutexDiesWhenRealtime) {
  std::shared_mutex Mutex{};
  Mutex.lock_shared();
  auto Func = [&]() { Mutex.unlock_shared(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

#endif // RADSAN_TEST_SHARED_MUTEX

TEST(TestRadsan, LaunchingAThreadDiesWhenRealtime) {
  auto Func = [&]() {
    std::thread Thread{[]() {}};
    Thread.join();
  };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

namespace {
void InvokeStdFunction(std::function<void()> &&function) { function(); }
} // namespace

TEST(TestRadsan, CopyingALambdaWithLargeCaptureDiesWhenRealtime) {
  std::array<float, 16> LotsOfData{};
  auto lambda = [LotsOfData]() mutable {
    // Stop everything getting optimised out
    LotsOfData[3] = 0.25f;
    EXPECT_EQ(16, LotsOfData.size());
    EXPECT_EQ(0.25f, LotsOfData[3]);
  };
  auto Func = [&]() { InvokeStdFunction(lambda); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, AccessingALargeAtomicVariableDiesWhenRealtime) {
  std::atomic<float> SmallAtomic{0.0f};
  ASSERT_TRUE(SmallAtomic.is_lock_free());
  RealtimeInvoke([&SmallAtomic]() { float x = SmallAtomic.load(); });

  std::atomic<std::array<float, 2048>> LargeAtomic{};
  ASSERT_FALSE(LargeAtomic.is_lock_free());
  auto Func = [&]() { auto x = LargeAtomic.load(); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, FirstCoutDiesWhenRealtime) {
  auto Func = []() { std::cout << "Hello, world!" << std::endl; };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, SecondCoutDiesWhenRealtime) {
  std::cout << "Hello, world";
  auto Func = []() { std::cout << "Hello, again!" << std::endl; };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, PrintfDiesWhenRealtime) {
  auto Func = []() { printf("Hello, world!\n"); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, ThrowingAnExceptionDiesWhenRealtime) {
  auto Func = [&]() {
    try {
      throw std::exception();
    } catch (std::exception &) {
    }
  };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsan, DoesNotDieIfTurnedOff) {
  std::mutex Mutex{};
  auto RealtimeUnsafeFunc = [&]() {
    radsan_off();
    Mutex.lock();
    Mutex.unlock();
    radsan_on();
  };
  RealtimeInvoke(RealtimeUnsafeFunc);
}
