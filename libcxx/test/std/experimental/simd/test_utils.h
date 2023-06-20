//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_UTIL_H
#define TEST_UTIL_H

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

using arithmetic_no_bool_types = types::concatenate_t<types::integer_types, types::floating_point_types>;

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

#endif // TEST_UTIL_H
