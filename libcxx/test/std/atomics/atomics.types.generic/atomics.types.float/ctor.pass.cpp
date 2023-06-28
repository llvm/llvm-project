//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr atomic() noexcept;
// constexpr atomic(floating-point-type) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>

#include "test_macros.h"

template <class T>
void test() {
  // constexpr atomic() noexcept;
  {
    constexpr std::atomic<T> a = {};
    assert(a.load() == T(0));
  }

  // constexpr atomic(floating-point-type) noexcept;
  {
    constexpr std::atomic<T> a = T(5.2);
    assert(a.load() == T(5.2));
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  test<long double>();
  return 0;
}
