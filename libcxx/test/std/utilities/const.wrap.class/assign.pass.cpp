//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// template<constexpr-param R>
//   constexpr auto operator=(R) const noexcept
//     -> constant_wrapper<value = R::value> { return {}; }

#include <concepts>
#include <type_traits>
#include <utility>

#include "helpers.h"

struct WithOps {
  int value;

  constexpr WithOps(int v) : value(v) {}

  constexpr WithOps operator=(int i) const { return WithOps{value + i}; }
};

struct OpsReturnNonStructural {
  int value;

  constexpr OpsReturnNonStructural(int v) : value(v) {}

  constexpr NonStructural operator=(int i) const { return NonStructural{value + i}; }
};

template <class T, class R>
concept HasAssign = requires(const T t, R r) {
  { t = r };
};

template <class T, class R>
concept HasNoexceptAssign = requires(const T t, R r) {
  { t = r } noexcept;
};

static_assert(!HasAssign<std::constant_wrapper<5>, std::constant_wrapper<3>>);
static_assert(!HasNoexceptAssign<std::constant_wrapper<5>, std::constant_wrapper<3>>);

static_assert(HasAssign<std::constant_wrapper<WithOps{5}>, std::constant_wrapper<3>>);
static_assert(HasNoexceptAssign<std::constant_wrapper<WithOps{5}>, std::constant_wrapper<3>>);

static_assert(!HasAssign<std::constant_wrapper<OpsReturnNonStructural{5}>, std::constant_wrapper<5>>);

constexpr bool test() {
  {
    // WithOps assignment
    const std::constant_wrapper<WithOps{5}> cwOps5;
    std::constant_wrapper<3> cw3;

    std::same_as<std::constant_wrapper<WithOps{8}>> decltype(auto) result = cwOps5 = cw3;
    static_assert(result.value.value == 8);
  }

  {
    // with integral_constant
    const std::constant_wrapper<WithOps{5}> cwOps5;
    std::integral_constant<int, 3> ic3;

    std::same_as<std::constant_wrapper<WithOps{8}>> decltype(auto) result = cwOps5 = ic3;
    static_assert(result.value.value == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
