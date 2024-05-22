//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// UNSUPPORTED: !non-lockfree-atomics

//  T exchange(T, memory_order = memory_order::seq_cst) volatile noexcept;
//  T exchange(T, memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileExchange = requires(volatile std::atomic<T>& a, T t) { a.exchange(t); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileExchange<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>() = (T(0))));

  // exchange
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    std::same_as<T> decltype(auto) r = a.exchange(T(1.2), std::memory_order::relaxed);
    assert(a.load() == T(1.2));
    assert(r == T(3.1));
  }

  // memory_order::release
  {
    auto exchange = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) {
      x.exchange(new_val, std::memory_order::release);
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::acquire); };
    test_acquire_release<T, MaybeVolatile>(exchange, load);
  }

  // memory_order::seq_cst
  {
    auto exchange_no_arg     = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.exchange(new_val); };
    auto exchange_with_order = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) {
      x.exchange(new_val, std::memory_order::seq_cst);
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(); };
    test_seq_cst<T, MaybeVolatile>(exchange_no_arg, load);
    test_seq_cst<T, MaybeVolatile>(exchange_with_order, load);
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
  // TODO https://github.com/llvm/llvm-project/issues/47978
  // test<long double>();

  return 0;
}
