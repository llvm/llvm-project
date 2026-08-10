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
// template <class T> struct is_abi_tag;
// template <class T> inline constexpr bool ex::is_abi_tag_v = ex::is_abi_tag<T>::value;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t N>
struct CheckIsAbiTag {
  template <class SimdAbi>
  void operator()() {
    static_assert(ex::is_abi_tag<SimdAbi>::value);

    static_assert(!ex::is_abi_tag<T>::value);
    static_assert(!ex::is_abi_tag<ex::simd<T, SimdAbi>>::value);
    static_assert(!ex::is_abi_tag<ex::simd_mask<T, SimdAbi>>::value);

    static_assert(ex::is_abi_tag_v<SimdAbi>);

    static_assert(!ex::is_abi_tag_v<T>);
    static_assert(!ex::is_abi_tag_v<ex::simd<T, SimdAbi>>);
    static_assert(!ex::is_abi_tag_v<ex::simd_mask<T, SimdAbi>>);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckIsAbiTag>();
  return 0;
}