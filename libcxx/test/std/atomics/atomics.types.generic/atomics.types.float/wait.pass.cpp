//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing
// XFAIL: !has-64-bit-atomics

// void wait(T old, memory_order order = memory_order::seq_cst) const volatile noexcept;
// void wait(T old, memory_order order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include "make_test_thread.h"
#  include <thread>
#endif

template <class T>
concept HasVolatileWait = requires(volatile std::atomic<T>& a, T t) { a.wait(T()); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileWait<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().wait(T())));

  // wait with different value
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    a.wait(T(1.1), std::memory_order::relaxed);
  }

#ifndef TEST_HAS_NO_THREADS
  // equal at the beginning and changed later
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
      a.notify_all();
      t.join();
    }
  }
#endif

  // memory_order::acquire
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
    auto load  = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      x.wait(T(9999.999), std::memory_order::acquire);
      return result;
    };
    test_acquire_release<T, MaybeVolatile>(store, load);
  }

  // memory_order::seq_cst
  {
    auto store       = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val); };
    auto load_no_arg = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      x.wait(T(9999.999));
      return result;
    };
    auto load_with_order = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      x.wait(T(9999.999), std::memory_order::seq_cst);
      return result;
    };
    test_seq_cst<T, MaybeVolatile>(store, load_no_arg);
    test_seq_cst<T, MaybeVolatile>(store, load_with_order);
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
