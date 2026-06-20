//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// template<class IntType>
//   constexpr byte operator<<(byte b, IntType shift) noexcept;
// Constraints: is_integral_v<IntType> is true.

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "type_algorithms.h"

template <class, class = void>
constexpr bool can_shl_byte = false;
template <class T>
constexpr bool can_shl_byte<T, std::void_t<decltype(std::byte{} << T{})>> = true;

struct test_functor {
  template <class I>
  constexpr void operator()() const {
    static_assert(can_shl_byte<I>);
    static_assert(noexcept(std::byte{} << I{}));
    static_assert(std::is_same_v<decltype(std::byte{} << I{}), std::byte>);

    std::byte b1{1};
    std::byte b3{3};

    assert(std::to_integer<int>(b1 << I{1}) == 2);
    assert(std::to_integer<int>(b1 << I{2}) == 4);
    assert(std::to_integer<int>(b3 << I{4}) == 48);
    assert(std::to_integer<int>(b3 << I{6}) == 192);
  }
};

struct test_failing_functor {
  template <class T>
  constexpr void operator()() const {
    static_assert(!can_shl_byte<T>);
  }
};

constexpr void test_bool(std::byte b) {
  static_assert(noexcept(b << true));
  static_assert(std::is_same_v<decltype(b << true), std::byte>);

  assert(std::to_integer<int>(b << true) == static_cast<unsigned char>(std::to_integer<int>(b) << 1));
  assert(std::to_integer<int>(b << false) == std::to_integer<int>(b));
}

constexpr bool test() {
  types::for_each(types::integer_types{}, test_functor{});
  types::for_each(types::floating_point_types{}, test_failing_functor{});
  types::for_each(types::type_list<void*, std::nullptr_t, void, int()>{}, test_failing_functor{});

  test_bool(std::byte{0});
  test_bool(std::byte{1});
  test_bool(std::byte{127});
  test_bool(std::byte{128});
  test_bool(std::byte{129});
  test_bool(std::byte{255});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
