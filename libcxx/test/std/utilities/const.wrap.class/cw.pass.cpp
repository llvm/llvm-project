//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

//   template<cw-fixed-value X>
//    constexpr auto cw = constant_wrapper<X>{};

#include <cassert>
#include <concepts>
#include <utility>

struct S {
  int value;

  constexpr S(int v) : value(v) {}

  constexpr bool operator==(const S& other) const { return value == other.value; }
};

constexpr bool test() {
  {
    // int constant
    std::same_as<const std::constant_wrapper<42>> decltype(auto) cw_val = std::cw<42>;
    static_assert(cw_val == 42);
  }

  {
    // struct constant
    constexpr S s{13};
    std::same_as<const std::constant_wrapper<s>> decltype(auto) cw_val = std::cw<s>;
    static_assert(cw_val == s);
  }

  {
    // array constant
    constexpr int arr[] = {1, 2, 3};
    // gcc complains that cw_val is unused
    [[maybe_unused]] std::same_as<const std::constant_wrapper<arr>> decltype(auto) cw_val = std::cw<arr>;
    static_assert(cw_val[0] == 1);
    static_assert(cw_val[1] == 2);
    static_assert(cw_val[2] == 3);
  }

  {
    // string literals
    [[maybe_unused]] std::same_as<const std::constant_wrapper<"hello">> decltype(auto) cw_val = std::cw<"hello">;
    static_assert(cw_val[0] == 'h');
    static_assert(cw_val[1] == 'e');
    static_assert(cw_val[2] == 'l');
    static_assert(cw_val[3] == 'l');
    static_assert(cw_val[4] == 'o');
    static_assert(cw_val[5] == '\0');
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
