//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// XFAIL: libcpp-has-no-experimental-rcu
// XFAIL: availability-rcu-missing
// XFAIL: !has-128-bit-atomics
// REQUIRES: std-at-least-c++26

// void rcu_barrier(rcu_domain& dom = rcu_default_domain()) noexcept;

#include <atomic>
#include <cassert>
#include <chrono>
#include <latch>
#include <rcu>
#include <stop_token>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::rcu_barrier()));
static_assert(noexcept(std::rcu_barrier(std::rcu_default_domain())));

class TestClass1 {
  int& destruction_count_;

public:
  TestClass1(int& c) : destruction_count_(c) {}
  ~TestClass1() { ++destruction_count_; }
};

struct TestClass2 {
  std::atomic<int>& dtor_count_;

  TestClass2(std::atomic<int>& ctor_count, std::atomic<int>& dtor_count) : dtor_count_(dtor_count) {
    ctor_count.fetch_add(1, std::memory_order_relaxed);
  }

  ~TestClass2() { dtor_count_.fetch_add(1, std::memory_order_relaxed); }
};

int main(int, char**) {
  {
    // rcu_barrier();

    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    std::rcu_retire(ptr);
    std::rcu_barrier();
    assert(destruction_count == 1);
  }
  {
    // rcu_retire(domain);

    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    std::rcu_retire(ptr);
    std::rcu_barrier(std::rcu_default_domain());
    assert(destruction_count == 1);
  }
  {
    // different thread

    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    std::rcu_retire(ptr);

    auto t = support::make_test_jthread([&] {
      std::rcu_barrier();
      assert(destruction_count == 1);
    });
  }
  {
    // multiple collector threads
    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    std::rcu_retire(ptr);

    constexpr int num_collectors = 4;
    std::vector<std::jthread> collector_threads;
    for (int i = 0; i < num_collectors; ++i) {
      collector_threads.push_back(support::make_test_jthread([&] {
        std::rcu_barrier();
        assert(destruction_count == 1);
      }));
    }
  }

  {
    // multiple writer and collector threads
    constexpr int num_writers    = 3;
    constexpr int num_collectors = 3;

    std::atomic<int> ctor_count = 0;
    std::atomic<int> dtor_count = 0;

    std::vector<std::jthread> writer_threads;
    for (int i = 0; i < num_writers; ++i) {
      writer_threads.emplace_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          auto* ptr = new TestClass2(ctor_count, dtor_count);
          std::rcu_retire(ptr);
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
      }));
    }

    std::vector<std::jthread> collector_threads;
    for (int i = 0; i < num_collectors; ++i) {
      collector_threads.emplace_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          std::rcu_barrier();
          std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
      }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    writer_threads.clear();
    collector_threads.clear();
    assert(ctor_count.load() == dtor_count.load());
  }

  {
    // multiple reader, writer and collector threads
    constexpr int num_readers    = 2;
    constexpr int num_writers    = 2;
    constexpr int num_collectors = 2;

    std::atomic<int> ctor_count = 0;
    std::atomic<int> dtor_count = 0;

    std::vector<std::jthread> reader_threads;
    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          auto& dom = std::rcu_default_domain();
          dom.lock();
          std::this_thread::sleep_for(std::chrono::microseconds(5));
          dom.unlock();
        }
      }));
    }

    std::vector<std::jthread> writer_threads;
    for (int i = 0; i < num_writers; ++i) {
      writer_threads.emplace_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          auto* ptr = new TestClass2(ctor_count, dtor_count);
          std::rcu_retire(ptr);
          std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
      }));
    }

    std::vector<std::jthread> collector_threads;
    for (int i = 0; i < num_collectors; ++i) {
      collector_threads.emplace_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          std::rcu_barrier();
          std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
      }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    reader_threads.clear();
    writer_threads.clear();
    collector_threads.clear();
    assert(ctor_count.load() == dtor_count.load());
  }
  {
    // object destroyed on another thread's rcu_barrier.
    // this thread's rcu_barrier should still wait
    // until the destructor of its retired object is returned

    std::atomic_bool main_barrier_start(false);
    std::atomic_bool t2_barrier_start(false);
    std::atomic_bool deleter_started(false);
    std::atomic_bool can_delete(false);

    auto t2 = support::make_test_jthread([&] {
      t2_barrier_start.wait(false);
      std::rcu_barrier();
    });

    auto t3 = support::make_test_jthread([&] {
      deleter_started.wait(false);
      main_barrier_start.store(true);
      main_barrier_start.notify_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      can_delete.store(true);
      can_delete.notify_all();
    });

    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    std::rcu_retire(ptr, [&](TestClass1* obj) {
      deleter_started.store(true);
      deleter_started.notify_all();
      can_delete.wait(false);
      delete obj;
    });

    t2_barrier_start.store(true);
    t2_barrier_start.notify_all();

    main_barrier_start.wait(false);

    std::rcu_barrier();
    assert(destruction_count == 1);
  }

  return 0;
}
