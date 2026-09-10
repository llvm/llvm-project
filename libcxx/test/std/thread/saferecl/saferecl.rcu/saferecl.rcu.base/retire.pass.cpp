//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// void retire(D d = D(), rcu_domain& dom = rcu_default_domain()) noexcept;

#include <atomic>
#include <cassert>
#include <latch>
#include <rcu>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

class TestClass1 : public std::rcu_obj_base<TestClass1> {
  int& destruction_count_;

public:
  TestClass1(int& c) : destruction_count_(c) {}
  ~TestClass1() { ++destruction_count_; }
};

struct Deleter {
  int i = 0;
  template <class T>
  void operator()(T* ptr) const {
    ptr->i_ = i;
    delete ptr;
  }
};

struct CustomDeleter : std::rcu_obj_base<CustomDeleter, Deleter> {
  int& i_;
  int& destruction_count_;
  CustomDeleter(int& i, int& destruction_count) : i_(i), destruction_count_(destruction_count) {}
  ~CustomDeleter() { ++destruction_count_; }
};

// Test noexcept
template <class T, class... Args>
concept NothrowRetireable = requires(T t, Args... args) {
  { t.retire(args...) } noexcept;
};

static_assert(NothrowRetireable<TestClass1>);
static_assert(NothrowRetireable<CustomDeleter>);
static_assert(NothrowRetireable<CustomDeleter, Deleter>);
static_assert(NothrowRetireable<CustomDeleter, Deleter, std::rcu_domain>);

int main(int, char**) {
  {
    // retire();

    int destruction_count = 0;
    auto* ptr             = new TestClass1(destruction_count);
    ptr->retire();
    std::rcu_barrier();
    assert(destruction_count == 1);
  }

  {
    // retire(deleter);
    int i                 = 0;
    int destruction_count = 0;
    auto* ptr             = new CustomDeleter(i, destruction_count);
    ptr->retire(Deleter{42});
    std::rcu_barrier();
    assert(destruction_count == 1);
    assert(i == 42);
  }

  {
    // retire(deleter, domain);
    int i                 = 0;
    int destruction_count = 0;
    auto* ptr             = new CustomDeleter(i, destruction_count);
    ptr->retire(Deleter{42}, std::rcu_default_domain());
    std::rcu_barrier();
    assert(destruction_count == 1);
    assert(i == 42);
  }
  {
    // different thread
    std::latch unlock_latch(1);

    auto& dom = std::rcu_default_domain();
    dom.lock();

    auto t = support::make_test_jthread([&] {
      int destruction_count = 0;
      auto* ptr             = new TestClass1(destruction_count);
      ptr->retire();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

      // should not be destroyed yet as the main thread is still holding the lock
      assert(destruction_count == 0);

      unlock_latch.count_down();

      std::rcu_barrier();
      assert(destruction_count == 1);
    });

    unlock_latch.wait();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    dom.unlock();
  }
  {
    // multiple reader threads
    constexpr int num_readers = 4;
    std::latch unlock_latch(1);
    std::atomic<int> reader_locked = 0;

    std::vector<std::jthread> reader_threads;
    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&] {
        auto& dom = std::rcu_default_domain();
        dom.lock();
        ++reader_locked;
        unlock_latch.wait();

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        dom.unlock();
      });
    }

    auto writer_thread = support::make_test_jthread([&] {
      int destruction_count = 0;
      auto* ptr             = new TestClass1(destruction_count);
      ptr->retire();
      while (reader_locked.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      // should not be destroyed yet as the main thread is still holding the lock
      assert(destruction_count == 0);

      unlock_latch.count_down();

      std::rcu_barrier();
      assert(destruction_count == 1);
    });
  }
  {
    // multiple writer threads
    constexpr int num_writers = 4;
    std::latch unlock_latch(num_writers);

    auto& dom = std::rcu_default_domain();
    dom.lock();

    std::vector<std::jthread> writer_threads;
    for (int i = 0; i < num_writers; ++i) {
      writer_threads.emplace_back([&] {
        int destruction_count = 0;
        auto* ptr             = new TestClass1(destruction_count);
        ptr->retire();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // should not be destroyed yet as the main thread is still holding the lock
        assert(destruction_count == 0);

        unlock_latch.count_down();

        std::rcu_barrier();
        assert(destruction_count == 1);
      });
    }

    unlock_latch.wait();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    dom.unlock();
  }
  {
    // multiple reader and writer threa    constexpr int num_readers = 4;
    constexpr int num_readers = 4;
    constexpr int num_writers = 3;
    std::latch unlock_latch(num_writers);
    std::atomic<int> reader_locked = 0;

    std::vector<std::jthread> reader_threads;
    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&] {
        auto& dom = std::rcu_default_domain();
        dom.lock();
        ++reader_locked;
        unlock_latch.wait();

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        dom.unlock();
      });
    };

    std::vector<std::jthread> writer_threads;
    for (int i = 0; i < num_writers; ++i) {
      writer_threads.emplace_back([&] {
        int destruction_count = 0;
        auto* ptr             = new TestClass1(destruction_count);
        ptr->retire();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // should not be destroyed yet as the main thread is still holding the lock
        assert(destruction_count == 0);

        unlock_latch.count_down();

        std::rcu_barrier();
        assert(destruction_count == 1);
      });
    }
  }

  return 0;
}
