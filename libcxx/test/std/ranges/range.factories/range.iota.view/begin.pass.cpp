//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr iterator begin() const;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

template<class T>
constexpr void testType() {
  {
    std::ranges::iota_view<T> io(T(0));
    assert(*io.begin() == T(0));
  }
  {
    std::ranges::iota_view<T> io(T(10));
    assert(*io.begin() == T(10));
    assert(*std::move(io).begin() == T(10));
  }
  {
    const std::ranges::iota_view<T> io(T(0));
    assert(*io.begin() == T(0));
  }
  {
    const std::ranges::iota_view<T> io(T(10));
    assert(*io.begin() == T(10));
  }
}

constexpr bool test() {
  testType<SomeInt>();
  testType<long long>();
  testType<unsigned long long>();
  testType<signed long>();
  testType<unsigned long>();
  testType<int>();
  testType<unsigned>();
  testType<short>();
  testType<unsigned short>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
