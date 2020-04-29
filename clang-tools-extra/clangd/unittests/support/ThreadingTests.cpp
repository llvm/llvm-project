//===-- ThreadingTests.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Threading.h"
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
} // namespace clangd
} // namespace clang
