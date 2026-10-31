//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-value

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator,(L, R) noexcept = delete;

#include <cassert>
#include <concepts>
#include <utility>

struct WithOps {
  int value;

  constexpr WithOps(int v) : value(v) {}

  friend constexpr auto operator,(const WithOps& /*l*/, WithOps r) { return WithOps{r.value}; }
};

struct NoOps {};

template <class L, class R>
concept HasComma = requires(L l, R r) {
  { l, r };
};

// Comma operator is deleted for constant_wrapper operands
static_assert(!HasComma<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasComma<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(!HasComma<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);

// Mixed operands - one constant_wrapper, one runtime type (uses built-in operator)
static_assert(HasComma<std::constant_wrapper<42>, int>);
static_assert(HasComma<int, std::constant_wrapper<42>>);

constexpr bool test() {
  {
    // only mixed with runtime parameters
    std::constant_wrapper<42> cw42;
    int i                                     = 0;
    std::same_as<int&> decltype(auto) result1 = (cw42, i);
    assert(result1 == 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
