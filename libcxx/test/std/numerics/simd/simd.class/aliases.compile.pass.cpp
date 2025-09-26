//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

// REQUIRES: std-at-least-c++26

#include <simd>
#include <type_traits>

#include "type_algorithms.h"
#include "../utils.h"

namespace dp = std::datapar;

template <class T>
inline constexpr bool is_signed_integral_constant = false;

template <class T, T N>
inline constexpr bool is_signed_integral_constant<std::integral_constant<T, N>> = std::is_signed_v<T>;

template <class T>
constexpr void test() {
  { // check size deduction
    using simd_t = dp::simd<T>;
    static_assert(std::is_same_v<typename simd_t::value_type, T>);
    static_assert(
        std::is_same_v<typename simd_t::mask_type, dp::basic_simd_mask<sizeof(T), typename simd_t::abi_type>>);

    static_assert(is_signed_integral_constant<std::remove_const_t<decltype(simd_t::size)>>);
  }

  { // check a few explicit sizes
    simd_utils::test_sizes([]<int N>(std::integral_constant<int, N>) { static_assert(dp::simd<T, N>::size == N); });
  }
}

static_assert([] {
  types::for_each(types::vectorizable_types{}, []<class T> { test<T>(); });

  return true;
}());
