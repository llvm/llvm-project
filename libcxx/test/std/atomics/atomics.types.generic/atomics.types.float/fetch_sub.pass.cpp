//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) noexcept;
// floating-point-type fetch_sub(floating-point-type,
//                               memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_sub(floating-point-type,
//                               memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>

#include "test_macros.h"

template <class T>
void test() {
  // fetch_add
  {
    std::atomic<T> a(3.1);
    std::same_as<T> decltype(auto) r = a.fetch_add(1.2);
    assert(r == T(3.1));
    assert(a.load() == T(3.1) + T(1.2));
  }

  // fetch_sub
  {
    std::atomic<T> a(3.1);
    std::same_as<T> decltype(auto) r = a.fetch_sub(1.2);
    assert(r == T(3.1));
    assert(a.load() == T(3.1) - T(1.2));
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
