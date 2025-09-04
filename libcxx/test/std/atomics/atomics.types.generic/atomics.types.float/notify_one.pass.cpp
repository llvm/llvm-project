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

//  void notify_one() volatile noexcept;
//  void notify_one() noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileNotifyOne = requires(volatile std::atomic<T>& a, T t) { a.notify_one(); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileNotifyOne<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().notify_one()));

  // bug?? wait can also fail for long double ??
  // should x87 80bit long double work at all?
  if constexpr (!std::same_as<T, long double>) {
    for (auto i = 0; i < 100; ++i) {
      const T old = T(3.1);
      MaybeVolatile<std::atomic<T>> a(old);

      std::atomic_bool started = false;
      bool done                = false;

      auto t = support::make_test_thread([&a, &started, old, &done] {
        started.store(true, std::memory_order::relaxed);

        a.wait(old);

        // likely to fail if wait did not block
        assert(done);
      });

      while (!started.load(std::memory_order::relaxed)) {
        std::this_thread::yield();
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      done = true;
      a.store(T(9.9));
      a.notify_one();
      t.join();
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
