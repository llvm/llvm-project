//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// constexpr cw-fixed-value(T (&arr)[Extent]) noexcept;

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
    // int array construction
    // the conversion from int array to cw-fixed-value<int array> uses the constructor
    constexpr int arr[] = {1, 2, 3};
    std::constant_wrapper<arr> cw{};
    assert(cw.value[0] == 1);
    assert(cw.value[1] == 2);
    assert(cw.value[2] == 3);
  }

  {
    // struct array construction
    constexpr S s[] = {{1}, {2}, {3}};
    std::constant_wrapper<s> cw{};
    assert(cw.value[0] == S{1});
    assert(cw.value[1] == S{2});
    assert(cw.value[2] == S{3});
  }

  {
    // calling the constructor
    constexpr int arr[] = {1, 2, 3, 4, 5};
    constexpr cw_fixed_value<const int[5]> ci(arr);
    std::constant_wrapper<ci> cw;
    assert(cw.value[0] == 1);
    assert(cw.value[1] == 2);
    assert(cw.value[2] == 3);
    assert(cw.value[3] == 4);
    assert(cw.value[4] == 5);

    static_assert(noexcept(cw_fixed_value<const int[5]>{arr}));
  }

  {
    // the constructor is implicit
    constexpr int arr[]                       = {1, 2, 3, 4, 5};
    constexpr cw_fixed_value<const int[5]> ci = arr;
    std::constant_wrapper<ci> cw;
    assert(cw.value[0] == 1);
    assert(cw.value[1] == 2);
    assert(cw.value[2] == 3);
    assert(cw.value[3] == 4);
    assert(cw.value[4] == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
