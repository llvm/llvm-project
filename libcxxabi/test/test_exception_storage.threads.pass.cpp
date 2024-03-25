//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, no-threads

#include "test_macros.h"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "assert_macros.h"
#include "concat_macros.h"
#include "../src/cxa_exception.h"

static int threads_remaining;
static std::mutex threads_remaining_lock;
static std::condition_variable threads_remaining_cv;

static void thread_code(void*& globals) {
  std::thread::id thread_id = std::this_thread::get_id();
  (void)thread_id;

  globals = __cxxabiv1::__cxa_get_globals();
  TEST_REQUIRE(globals != nullptr,
               TEST_WRITE_CONCATENATED("Got null result from __cxa_get_globals on thread ", thread_id));

  void* fast_globals = __cxxabiv1::__cxa_get_globals_fast();
  TEST_REQUIRE(globals == fast_globals,
               TEST_WRITE_CONCATENATED("__cxa_get_globals returned ", globals, " but __cxa_get_globals_fast returned ",
                                       fast_globals, " on thread ", thread_id));

  // Ensure that all threads are running at the same time, since we check for
  // duplicate globals below. We do this manually instead of using std::barrier
  // or std::latch to avoid requiring C++20.
  std::unique_lock<std::mutex> lock(threads_remaining_lock);
  --threads_remaining;
  if (threads_remaining == 0) {
    lock.unlock();
    threads_remaining_cv.notify_all();
  } else {
    threads_remaining_cv.wait(lock, []() { return threads_remaining == 0; });
  }
}

int main(int, char**) {
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4; // arbitrary fallback value

  std::vector<void*> thread_globals(num_threads);
  std::vector<std::thread> threads;
  threads_remaining = num_threads;

  // Make the threads, let them run, and wait for them to finish
  for (int i = 0; i < num_threads; ++i)
    threads.emplace_back(thread_code, std::ref(thread_globals[i]));
  for (std::thread& thread : threads)
    thread.join();

  std::sort(thread_globals.begin(), thread_globals.end());
  for (int i = 1; i < num_threads; ++i) {
    TEST_REQUIRE(thread_globals[i - 1] != thread_globals[i],
                 TEST_WRITE_CONCATENATED("Duplicate thread globals ", thread_globals[i]));
  }

  return 0;
}
