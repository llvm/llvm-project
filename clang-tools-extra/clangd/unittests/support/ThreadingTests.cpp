//===-- ThreadingTests.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Threading.h"
#include "llvm/ADT/DenseMap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mutex>

namespace clang {
namespace clangd {
class ThreadingTest : public ::testing::Test {};

TEST_F(ThreadingTest, TaskRunner) {
  const int TasksCnt = 100;
  // This should be const, but MSVC does not allow to use const vars in lambdas
  // without capture. On the other hand, clang gives a warning that capture of
  // const var is not required.
  // Making it non-const makes both compilers happy.
  int IncrementsPerTask = 1000;

  std::mutex Mutex;
  int Counter(0); /* GUARDED_BY(Mutex) */
  {
    AsyncTaskRunner Tasks;
    auto scheduleIncrements = [&]() {
      for (int TaskI = 0; TaskI < TasksCnt; ++TaskI) {
        Tasks.runAsync("task", [&Counter, &Mutex, IncrementsPerTask]() {
          for (int Increment = 0; Increment < IncrementsPerTask; ++Increment) {
            std::lock_guard<std::mutex> Lock(Mutex);
            ++Counter;
          }
        });
      }
    };

    {
      // Make sure runAsync is not running tasks synchronously on the same
      // thread by locking the Mutex used for increments.
      std::lock_guard<std::mutex> Lock(Mutex);
      scheduleIncrements();
    }

    Tasks.wait();
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      ASSERT_EQ(Counter, TasksCnt * IncrementsPerTask);
    }

    {
      std::lock_guard<std::mutex> Lock(Mutex);
      Counter = 0;
      scheduleIncrements();
    }
  }
  // Check that destructor has waited for tasks to finish.
  std::lock_guard<std::mutex> Lock(Mutex);
  ASSERT_EQ(Counter, TasksCnt * IncrementsPerTask);
}

TEST_F(ThreadingTest, Memoize) {
  const unsigned NumThreads = 5;
  const unsigned NumKeys = 100;
  const unsigned NumIterations = 100;

  Memoize<llvm::DenseMap<int, int>> Cache;
  std::atomic<unsigned> ComputeCount(0);
  std::atomic<int> ComputeResult[NumKeys];
  std::fill(std::begin(ComputeResult), std::end(ComputeResult), -1);

  AsyncTaskRunner Tasks;
  for (unsigned I = 0; I < NumThreads; ++I)
    Tasks.runAsync("worker" + std::to_string(I), [&] {
      for (unsigned J = 0; J < NumIterations; J++)
        for (unsigned K = 0; K < NumKeys; K++) {
          int Result = Cache.get(K, [&] { return ++ComputeCount; });
          EXPECT_THAT(ComputeResult[K].exchange(Result),
                      testing::AnyOf(-1, Result))
              << "Got inconsistent results from memoize";
        }
    });
  Tasks.wait();
  EXPECT_GE(ComputeCount, NumKeys) << "Computed each key once";
  EXPECT_LE(ComputeCount, NumThreads * NumKeys)
      << "Worst case, computed each key in every thread";
  for (int Result : ComputeResult)
    EXPECT_GT(Result, 0) << "All results in expected domain";
}

TEST_F(ThreadingTest, MemoizeDeterministic) {
  Memoize<llvm::DenseMap<int, char>> Cache;

  // Spawn two parallel computations, A and B.
  // Force concurrency: neither can finish until both have started.
  // Verify that cache returns consistent results.
  AsyncTaskRunner Tasks;
  std::atomic<char> ValueA(0), ValueB(0);
  Notification ReleaseA, ReleaseB;
  Tasks.runAsync("A", [&] {
    ValueA = Cache.get(0, [&] {
      ReleaseB.notify();
      ReleaseA.wait();
      return 'A';
    });
  });
  Tasks.runAsync("A", [&] {
    ValueB = Cache.get(0, [&] {
      ReleaseA.notify();
      ReleaseB.wait();
      return 'B';
    });
  });
  Tasks.wait();

  ASSERT_EQ(ValueA, ValueB);
  ASSERT_THAT(ValueA.load(), testing::AnyOf('A', 'B'));
}

} // namespace clangd
} // namespace clang
