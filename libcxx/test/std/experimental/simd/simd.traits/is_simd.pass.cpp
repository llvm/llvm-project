//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.traits]
// template <class T> struct ex::is_simd;
// template <class T> inline constexpr bool ex::is_simd_v =
// ex::is_simd<T>::value;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckIsSimd {
  template <class SimdAbi>
  void operator()() {
    static_assert(ex::is_simd<ex::simd<T, SimdAbi>>::value);

    static_assert(!ex::is_simd<T>::value);
    static_assert(!ex::is_simd<ex::simd_mask<T, SimdAbi>>::value);

    static_assert(ex::is_simd_v<ex::simd<T, SimdAbi>>);

    static_assert(!ex::is_simd_v<T>);
    static_assert(!ex::is_simd_v<ex::simd_mask<T, SimdAbi>>);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckIsSimd>();
  return 0;
}
