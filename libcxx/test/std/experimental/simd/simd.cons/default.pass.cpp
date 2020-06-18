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
// [simd.class]
// simd() = default;

#include <cstdint>
#include <experimental/simd>

#include "test_macros.h"

namespace ex = std::experimental::parallelism_v2;

int main(int, char**) {
  static_assert(ex::native_simd<int32_t>().size() > 0, "");
  static_assert(ex::fixed_size_simd<int32_t, 4>().size() == 4, "");
  static_assert(ex::fixed_size_simd<int32_t, 5>().size() == 5, "");
  static_assert(ex::fixed_size_simd<int32_t, 1>().size() == 1, "");
  static_assert(ex::fixed_size_simd<char, 32>().size() == 32, "");

  return 0;
}
