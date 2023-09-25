//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>
#include <utility>
#include <experimental/simd>
#include "type_algorithms.h"

namespace ex = std::experimental::parallelism_v2;

constexpr std::size_t max_simd_size = 32;

template <template <class T, std::size_t N> class F>
struct TestAllSimdAbiFunctor {
  template <class T, std::size_t N>
  using sized_abis = types::type_list<ex::simd_abi::fixed_size<N>, ex::simd_abi::deduce_t<T, N>>;

  template <class T, std::size_t... Ns>
  void instantiate_with_n(std::index_sequence<Ns...>) {
    (types::for_each(sized_abis<T, Ns + 1>{}, F<T, Ns + 1>{}), ...);
  }

  template <class T>
  void operator()() {
    using abis = types::type_list<ex::simd_abi::scalar, ex::simd_abi::native<T>, ex::simd_abi::compatible<T>>;
    types::for_each(abis{}, F<T, 1>());

    instantiate_with_n<T>(std::make_index_sequence<max_simd_size - 1>{});
  }
};

// TODO: Support long double (12 bytes) for 32-bits x86
#ifdef __i386__
using arithmetic_no_bool_types = types::concatenate_t<types::integer_types, types::type_list<float, double>>;
#else
using arithmetic_no_bool_types = types::concatenate_t<types::integer_types, types::floating_point_types>;
#endif

template <template <class T, std::size_t N> class Func>
void test_all_simd_abi() {
  types::for_each(arithmetic_no_bool_types(), TestAllSimdAbiFunctor<Func>());
}

constexpr size_t bit_ceil(size_t val) {
  size_t pow = 1;
  while (pow < val)
    pow <<= 1;
  return pow;
}

template <class From, class To, class = void>
inline constexpr bool is_non_narrowing_convertible_v = false;

template <class From, class To>
inline constexpr bool is_non_narrowing_convertible_v<From, To, std::void_t<decltype(To{std::declval<From>()})>> = true;

template <std::size_t ArraySize, class SimdAbi, class T, class U = T>
void assert_simd_values_equal(const ex::simd<T, SimdAbi>& origin_simd, const std::array<U, ArraySize>& expected_value) {
  for (std::size_t i = 0; i < origin_simd.size(); ++i)
    assert(origin_simd[i] == static_cast<T>(expected_value[i]));
}

template <std::size_t ArraySize, class T, class SimdAbi>
void assert_simd_mask_values_equal(const ex::simd_mask<T, SimdAbi>& origin_mask,
                                   const std::array<bool, ArraySize>& expected_value) {
  for (std::size_t i = 0; i < origin_mask.size(); ++i)
    assert(origin_mask[i] == expected_value[i]);
}

#endif // TEST_UTIL_H
