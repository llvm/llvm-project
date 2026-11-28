//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <variant>
#include <simd>
#include <type_traits>

#include "type_algorithms.h"
#include "../../utils.h"

namespace dp = std::datapar;

template <class SimdT, class Arg>
concept has_broadcast_constructor = requires { dp::simd<SimdT>(std::declval<Arg>()); };

static_assert(has_broadcast_constructor<int32_t, int16_t>);
static_assert(has_broadcast_constructor<double, float>);
static_assert(!has_broadcast_constructor<float, double>);
static_assert(!has_broadcast_constructor<int16_t, int32_t>);
static_assert(!has_broadcast_constructor<int8_t, int32_t>);
static_assert(!has_broadcast_constructor<uint32_t, int32_t>);
static_assert(!has_broadcast_constructor<float, int32_t>);
static_assert(!has_broadcast_constructor<int32_t, float>);

struct convertible_to_int {
  operator int();
};

static_assert(has_broadcast_constructor<int, convertible_to_int>);
static_assert(has_broadcast_constructor<float, convertible_to_int>);

struct almost_constexpr_wrapper_like {
  static constexpr int value = 34;

  constexpr int operator()() const { return value; }
  operator int() const { return value; }
  friend constexpr bool operator==(almost_constexpr_wrapper_like, almost_constexpr_wrapper_like) = default;
  friend constexpr bool operator==(almost_constexpr_wrapper_like lhs, int rhs) { return lhs.value == rhs; }
};
LIBCPP_STATIC_ASSERT(!dp::__constexpr_wrapper_like<almost_constexpr_wrapper_like>);

static_assert(has_broadcast_constructor<int, almost_constexpr_wrapper_like>);
static_assert(has_broadcast_constructor<float, almost_constexpr_wrapper_like>);

template <class T, T Val>
struct constexpr_wrapper_like {
  static constexpr T value = Val;

  constexpr T operator()() const { return value; }
  constexpr operator T() const { return value; }
  friend constexpr bool operator==(constexpr_wrapper_like, constexpr_wrapper_like) = default;
  friend constexpr bool operator==(constexpr_wrapper_like lhs, T rhs) { return lhs.value == rhs; }
};
LIBCPP_STATIC_ASSERT(dp::__constexpr_wrapper_like<constexpr_wrapper_like<int, 1>>);

static_assert(!has_broadcast_constructor<int, constexpr_wrapper_like<std::monostate, std::monostate{}>>);
static_assert(!has_broadcast_constructor<int, constexpr_wrapper_like<float, 3.4f>>);
static_assert(has_broadcast_constructor<int, constexpr_wrapper_like<float, 3.f>>);
static_assert(has_broadcast_constructor<float, constexpr_wrapper_like<float, 3.4f>>);

template <class T>
constexpr void test() {
  simd_utils::test_sizes([]<int N>(std::integral_constant<int, N>) {
    dp::simd<T, N> vec(T(1));
    for (auto i = 0; i != vec.size(); ++i)
      assert(vec[i] == T(1));
  });
}

constexpr bool test() {
  types::for_each(types::vectorizable_types{}, []<class T> { test<T>(); });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
