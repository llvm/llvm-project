//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_NUMERICS_SIMD_UTILS_H
#define TEST_STD_NUMERICS_SIMD_UTILS_H

#include <array>
#include <simd>
#include <type_traits>

#include "type_algorithms.h"

namespace types {
using vectorizable_float_types = type_list<float, double>;
using vectorizable_types       = types::concatenate_t<types::standard_integer_types, vectorizable_float_types>;
} // namespace types

namespace simd_utils {
template <class Func>
constexpr void test_sizes(Func f) {
  f(std::integral_constant<int, 1>{});
  f(std::integral_constant<int, 2>{});
  f(std::integral_constant<int, 3>{});
  f(std::integral_constant<int, 4>{});
  f(std::integral_constant<int, 8>{});
  f(std::integral_constant<int, 15>{});
  f(std::integral_constant<int, 16>{});
  f(std::integral_constant<int, 17>{});
  f(std::integral_constant<int, 32>{});
  f(std::integral_constant<int, 64>{});
}

template <std::size_t N>
constexpr std::datapar::simd_mask<int, N> make_mask(std::array<bool, N> bools) {
  std::array<int, N> bools_as_int;
  for (size_t i = 0; i != N; ++i)
    bools_as_int[i] = bools[i] ? 1 : 0;
  return std::datapar::simd<int, N>(1) == std::datapar::simd<int, N>(bools_as_int);
}
} // namespace simd_utils

#endif // TEST_STD_NUMERICS_SIMD_UTILS_H
