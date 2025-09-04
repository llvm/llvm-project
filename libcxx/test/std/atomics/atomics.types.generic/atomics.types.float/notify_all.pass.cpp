//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// XFAIL: availability-synchronization_library-missing
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

//  void notify_all() volatile noexcept;
//  void notify_all() noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileNotifyAll = requires(volatile std::atomic<T>& a, T t) { a.notify_all(); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileNotifyAll<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().notify_all()));

  // bug?? wait can also fail for long double ??
  // should x87 80bit long double work at all?
  if constexpr (!std::same_as<T, long double>) {
    for (auto i = 0; i < 100; ++i) {
      const T old = T(3.1);
      MaybeVolatile<std::atomic<T>> a(old);

      bool done                      = false;
      std::atomic<int> started_num   = 0;
      std::atomic<int> wait_done_num = 0;

      constexpr auto number_of_threads = 8;
      std::vector<std::thread> threads;
      threads.reserve(number_of_threads);

      for (auto j = 0; j < number_of_threads; ++j) {
        threads.push_back(support::make_test_thread([&a, &started_num, old, &done, &wait_done_num] {
          started_num.fetch_add(1, std::memory_order::relaxed);

          a.wait(old);
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
      a.store(T(9.9));
      a.notify_all();

      // notify_all should unblock all the threads so that the loop below won't stuck
      while (wait_done_num.load(std::memory_order::relaxed) != number_of_threads) {
        std::this_thread::yield();
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }
  }
}

template <class T>
void test() {
  test_impl<T>();
  if constexpr (std::atomic<T>::is_always_lock_free) {
    test_impl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  // TODO https://llvm.org/PR48634
  // test<long double>();

  return 0;
}
