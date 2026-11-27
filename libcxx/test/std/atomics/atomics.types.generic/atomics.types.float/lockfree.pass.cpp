//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

//   static constexpr bool is_always_lock_free = implementation-defined;
//   bool is_lock_free() const volatile noexcept;
//   bool is_lock_free() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept isLockFreeNoexcept = requires(T t) {
  { t.is_lock_free() } noexcept;
};

template <class T>
void test() {
  static_assert(isLockFreeNoexcept<const std::atomic<T>&>);
  static_assert(isLockFreeNoexcept<const volatile std::atomic<T>&>);

  //   static constexpr bool is_always_lock_free = implementation-defined;
  { [[maybe_unused]] constexpr std::same_as<const bool> decltype(auto) r = std::atomic<T>::is_always_lock_free; }

  //   bool is_lock_free() const volatile noexcept;
  {
    const volatile std::atomic<T> a;
    std::same_as<bool> decltype(auto) r = a.is_lock_free();
    if (std::atomic<T>::is_always_lock_free) {
      assert(r);
    }
  }

  //   bool is_lock_free() const noexcept;
  {
    const std::atomic<T> a;
    std::same_as<bool> decltype(auto) r = a.is_lock_free();
    if (std::atomic<T>::is_always_lock_free) {
      assert(r);
    }
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  // TODO https://llvm.org/PR48634
  // test<long double>();

  return 0;
}
