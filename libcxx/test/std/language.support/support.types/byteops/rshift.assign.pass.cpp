//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// template<class IntType>
//   constexpr byte& operator>>=(byte& b, IntType shift) noexcept;
// Constraints: is_integral_v<IntType> is true.

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "type_algorithms.h"

template <class, class = void>
constexpr bool can_shr_assign_byte = false;
template <class T>
constexpr bool can_shr_assign_byte<T, std::void_t<decltype(std::declval<std::byte&>() >>= std::declval<T>())>> = true;

template <class I>
constexpr std::byte test_op(std::byte b) {
  static_assert(noexcept(b >>= I{2}));
  static_assert(std::is_same_v<decltype(b >>= I{2}), std::byte&>);

  std::byte& ret = b >>= I{2};
  assert(&ret == &b);
  return ret;
}

struct test_functor {
  template <class I>
  constexpr void operator()() const {
    static_assert(can_shr_assign_byte<I>);

    std::byte b16{16};
    std::byte b192{192};

    assert(std::to_integer<int>(test_op<I>(b16)) == 4);
    assert(std::to_integer<int>(test_op<I>(b192)) == 48);
  }
};

struct test_failing_functor {
  template <class T>
  constexpr void operator()() const {
    static_assert(!can_shr_assign_byte<T>);
  }
};

constexpr void test_bool(std::byte b) {
  static_assert(noexcept(b >>= true));
  static_assert(std::is_same_v<decltype(b >>= true), std::byte&>);

  {
    std::byte b1   = b;
    std::byte& ret = b1 >>= true;
    assert(&ret == &b1);
    assert(std::to_integer<int>(b1) == std::to_integer<int>(b) >> 1);
  }
  {
    std::byte b1   = b;
    std::byte& ret = b1 >>= false;
    assert(&ret == &b1);
    assert(std::to_integer<int>(b1) == std::to_integer<int>(b));
  }
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
