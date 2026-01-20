//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// void notify_all() const noexcept;

#include <atomic>
#include <cassert>
#include <thread>
#include <type_traits>
#include <vector>

#include "atomic_helpers.h"
#include "make_test_thread.h"
#include "test_macros.h"

template <typename T>
struct TestNotifyAll {
  void operator()() const {
    T x(T(1));
    std::atomic_ref<T> const a(x);

    bool done                      = false;
    std::atomic<int> started_num   = 0;
    std::atomic<int> wait_done_num = 0;

    constexpr auto number_of_threads = 8;
    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);

    for (auto j = 0; j < number_of_threads; ++j) {
      threads.push_back(support::make_test_thread([&a, &started_num, &done, &wait_done_num] {
        started_num.fetch_add(1, std::memory_order::relaxed);

        a.wait(T(1));
        wait_done_num.fetch_add(1, std::memory_order::relaxed);

        // likely to fail if wait did not block
        assert(done);
      }));
    }

    while (started_num.load(std::memory_order::relaxed) != number_of_threads) {
      std::this_thread::yield();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    done = true;
    a.store(T(3));
    a.notify_all();

    // notify_all should unblock all the threads so that the loop below won't stuck
    while (wait_done_num.load(std::memory_order::relaxed) != number_of_threads) {
      std::this_thread::yield();
    }

    for (auto& thread : threads) {
      thread.join();
    }

    ASSERT_NOEXCEPT(a.notify_all());
  }
};

int main(int, char**) {
  TestEachAtomicType<TestNotifyAll>()();
  return 0;
}
