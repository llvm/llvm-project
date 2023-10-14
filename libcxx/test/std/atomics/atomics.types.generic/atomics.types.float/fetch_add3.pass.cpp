//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS(has-latomic): -latomic

// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileFetchAdd = requires(volatile std::atomic<T>& a, T t) { a.fetch_add(t); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void testImpl() {
  static_assert(HasVolatileFetchAdd<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().fetch_add(T(0))));
  // memory_order::release
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      x.fetch_add(new_val - old_val, std::memory_order::release);
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::acquire); };
    test_acquire_release<T, MaybeVolatile>(store, load);
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
