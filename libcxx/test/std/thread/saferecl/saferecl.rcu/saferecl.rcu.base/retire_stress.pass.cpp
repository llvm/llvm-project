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
#include <rcu>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct TestClass : std::rcu_obj_base<TestClass> {
  size_t index_;

  TestClass(size_t i) : index_(i) {}
  TestClass(const TestClass&)            = delete;
  TestClass(TestClass&&)                 = delete;
  TestClass& operator=(const TestClass&) = delete;
  TestClass& operator=(TestClass&&)      = delete;
  ~TestClass();
};

struct ObjectStore {
  static constexpr size_t max_objects                          = 10000;
  static inline std::atomic<size_t> next_index                 = 0;
  static inline std::vector<std::atomic_bool> object_destroyed = std::vector<std::atomic_bool>(max_objects);

  static TestClass* create() {
    size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
    return index < max_objects ? new TestClass(index) : nullptr;
  }

  static bool is_destroyed(size_t index) {
    return ObjectStore::object_destroyed[index].load(std::memory_order_relaxed);
  }
};

TestClass::~TestClass() {
  assert(!ObjectStore::is_destroyed(index_));
  ObjectStore::object_destroyed[index_].store(true, std::memory_order_relaxed);
}

int main(int, char**) {
  {
    // stress test
    constexpr int num_readers = 4;
    constexpr int num_writers = 3;

    std::atomic<TestClass*> global_ptr = ObjectStore::create();

    std::vector<std::jthread> reader_threads;
    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&](std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
          auto& dom = std::rcu_default_domain();
          dom.lock();
          auto ptr = global_ptr.load();
          assert(!ObjectStore::is_destroyed(ptr->index_));
          dom.unlock();
        }
      });
    }

    std::vector<std::jthread> writer_threads;
    for (int i = 0; i < num_writers; ++i) {
      writer_threads.emplace_back([&]() {
        TestClass* ptr = ObjectStore::create();
        for (; ptr != nullptr; ptr = ObjectStore::create()) {
          auto old      = global_ptr.exchange(ptr);
          auto oldIndex = old->index_;
          old->retire();
          std::rcu_barrier();

          assert(ObjectStore::is_destroyed(oldIndex));
        }
      });
    }
  }

  return 0;
}
