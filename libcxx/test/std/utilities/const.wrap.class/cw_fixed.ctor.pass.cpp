//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// constexpr cw-fixed-value(type v) noexcept : data(v) {}

#include <cassert>
#include <utility>

template <auto v>
auto helper(std::constant_wrapper<v>) -> decltype(v);

template <class T>
using cw_fixed_value = decltype(helper(std::constant_wrapper<T{}>{}));

struct S {
  int value;

  constexpr S(int v = 0) : value(v) {}

  constexpr bool operator==(const S& other) const { return value == other.value; }
};

constexpr bool test() {
  {
    // int construction
    // the conversion from int to cw-fixed-value<int> uses the constructor
    std::constant_wrapper<42> cw{};
    assert(cw.value == 42);
  }

  {
    // struct construction
    std::constant_wrapper<S{13}> cw{};
    assert(cw.value == S{13});
  }

  {
    // calling the constructor
    constexpr cw_fixed_value<int> ci{42};
    std::constant_wrapper<ci> cw;
    assert(cw == 42);

    static_assert(noexcept(cw_fixed_value<int>{42}));
  }

  {
    // the constructor is implicit
    constexpr cw_fixed_value<int> ci = 42;
    std::constant_wrapper<ci> cw;
    assert(cw == 42);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
