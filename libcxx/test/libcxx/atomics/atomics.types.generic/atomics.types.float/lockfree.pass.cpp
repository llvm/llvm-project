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

#include "test_macros.h"

template <class T>
void test() {
  //   static constexpr bool is_always_lock_free = implementation-defined;
  {
    bool r = std::atomic<T>::is_always_lock_free;
    assert(r == __atomic_always_lock_free(sizeof(std::__cxx_atomic_impl<T>), 0));
  }

  //   bool is_lock_free() const volatile noexcept;
  {
    const volatile std::atomic<T> a;
    bool r = a.is_lock_free();
    assert(r == __cxx_atomic_is_lock_free(sizeof(std::__cxx_atomic_impl<T>)));
  }

  //   bool is_lock_free() const noexcept;
  {
    const std::atomic<T> a;
    bool r = a.is_lock_free();
    assert(r == __cxx_atomic_is_lock_free(sizeof(std::__cxx_atomic_impl<T>)));
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  // TODO https://llvm.org/PR48634
  // test<long double>();

  return 0;
}
