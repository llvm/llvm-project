//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <experimental/simd>
//
// [simd.casts]
// template <class T, class U, class Abi> see below ex::simd_cast<(const
// ex::simd<U, Abi>&);

#include <experimental/simd>
#include <cstdint>

namespace ex = std::experimental::parallelism_v2;

static_assert(
    std::is_same<decltype(ex::simd_cast<int32_t>(ex::native_simd<int32_t>())),
                 ex::native_simd<int32_t>>::value,
    "");

static_assert(std::is_same<decltype(ex::simd_cast<int64_t>(
                               ex::fixed_size_simd<int32_t, 4>())),
                           ex::fixed_size_simd<int64_t, 4>>::value,
              "");

static_assert(
    std::is_same<decltype(ex::simd_cast<ex::fixed_size_simd<int64_t, 1>>(
                     ex::simd<int32_t, ex::simd_abi::scalar>())),
                 ex::fixed_size_simd<int64_t, 1>>::value,
    "");

static_assert(
    std::is_same<
        decltype(ex::simd_cast<ex::simd<int64_t, ex::simd_abi::scalar>>(
            ex::fixed_size_simd<int32_t, 1>())),
        ex::simd<int64_t, ex::simd_abi::scalar>>::value,
    "");

int main(int, char**) {
  return 0;
}
