//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// void wait(T old, memory_order order = memory_order::seq_cst) const volatile noexcept;
// void wait(T old, memory_order order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileWait = requires(volatile std::atomic<T> a, T t) { a.wait(T()); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void testImpl() {
  static_assert(HasVolatileWait<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().wait(T())));

  // wait different value
  {
    MaybeVolatile<std::atomic<T>> a(3.1);
    a.wait(T(1.1), std::memory_order::relaxed);
  }

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
  testImpl<T>();
  if constexpr (std::atomic<T>::is_always_lock_free) {
    testImpl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
