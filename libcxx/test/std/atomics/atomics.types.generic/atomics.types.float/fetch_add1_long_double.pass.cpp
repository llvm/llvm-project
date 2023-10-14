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

  // fetch_add
  {
    MaybeVolatile<std::atomic<T>> a(3.1);
    std::same_as<T> decltype(auto) r = a.fetch_add(T(1.2), std::memory_order::relaxed);
    assert(r == T(3.1));
    assert(a.load() == T(3.1) + T(1.2));
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
  test<long double>();

  return 0;
}
