//===-- ThreadingTests.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Threading.h"
#include "gtest/gtest.h"
#include <mutex>

namespace clang {
namespace clangd {
class ThreadingTest : public ::testing::Test {};

TEST_F(ThreadingTest, TaskRunner) {
  const int TasksCnt = 100;
  const int IncrementsPerTask = 1000;

  std::mutex Mutex;
  int Counter(0); /* GUARDED_BY(Mutex) */
  {
    AsyncTaskRunner Tasks;
    auto scheduleIncrements = [&]() {
      for (int TaskI = 0; TaskI < TasksCnt; ++TaskI) {
        Tasks.runAsync([&Counter, &Mutex]() {
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

    Tasks.waitForAll();
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
