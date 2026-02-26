//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: hexagon-registered-target
// ADDITIONAL_COMPILE_FLAGS: -mhvx

// <experimental/simd>
//
// [simd.traits]
// Verify that native_simd sizes match the 128-byte HVX vector register
// width.  This mirrors the checks in simd_size.pass.cpp for the native
// ABI but with concrete HVX expectations.

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

static_assert(_LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES == 128, "HVX native SIMD width must be 128 bytes");

struct CheckHvxNativeSimdSize {
  template <class T>
  void operator()() {
    constexpr std::size_t expected = 128 / sizeof(T);

    static_assert(ex::simd_size_v<T, ex::simd_abi::native<T>> == expected);
    static_assert(ex::simd_size<T, ex::simd_abi::native<T>>::value == expected);
    static_assert(ex::native_simd<T>::size() == expected);
    static_assert(ex::is_abi_tag_v<ex::simd_abi::native<T>>);

    // memory_alignment for native should be 128 bytes (one HVX register)
    if constexpr (!std::is_same_v<T, long double>)
      static_assert(ex::memory_alignment_v<ex::native_simd<T>> == 128);
  }
};

int main(int, char**) {
  types::for_each(arithmetic_no_bool_types(), CheckHvxNativeSimdSize());
  return 0;
}
