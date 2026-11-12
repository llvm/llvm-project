//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// template<class IntType>
//   constexpr IntType to_integer(byte b) noexcept;
// Constraints: is_integral_v<IntType> is true.

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "type_algorithms.h"

template <class, class = void>
constexpr bool can_to_integer = false;
template <class T>
constexpr bool can_to_integer<T, std::void_t<decltype(std::to_integer<T>(std::byte{}))>> = true;

struct test_functor {
  template <class I>
  constexpr void operator()() const {
    static_assert(can_to_integer<I>);
    static_assert(noexcept(std::to_integer<I>(std::byte{})));
    static_assert(std::is_same_v<decltype(std::to_integer<I>(std::byte{})), I>);

    std::byte b1{1};
    std::byte b3{3};

    assert(std::to_integer<I>(b1) == I{1});
    assert(std::to_integer<I>(b3) == I{3});
  }
};

struct test_failing_functor {
  template <class T>
  constexpr void operator()() const {
    static_assert(!can_to_integer<T>);
  }
};

constexpr bool test() {
  types::for_each(types::integer_types{}, test_functor{});
  types::for_each(types::floating_point_types{}, test_failing_functor{});
  types::for_each(types::type_list<void*, std::nullptr_t, void, int()>{}, test_failing_functor{});

  static_assert(noexcept(std::to_integer<bool>(std::byte{})));
  static_assert(std::is_same_v<decltype(std::to_integer<bool>(std::byte{})), bool>);

  assert(!std::to_integer<bool>(std::byte{0}));
  assert(std::to_integer<bool>(std::byte{1}));
  assert(std::to_integer<bool>(std::byte{127}));
  assert(std::to_integer<bool>(std::byte{128}));
  assert(std::to_integer<bool>(std::byte{129}));
  assert(std::to_integer<bool>(std::byte{255}));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
