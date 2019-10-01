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
// [simd.traits]
// template <class T, size_t N> struct abi_for_size { using type = see below ;
// }; template <class T, size_t N> using ex::abi_for_size_t = typename
// ex::abi_for_size<T, N>::type;

#include <cstdint>
#include <experimental/simd>

#include "test_macros.h"

namespace ex = std::experimental::parallelism_v2;

static_assert(std::is_same<typename ex::abi_for_size<int, 4>::type,
                           ex::simd_abi::fixed_size<4>>::value,
              "");

static_assert(std::is_same<ex::abi_for_size_t<int, 4>,
                           ex::simd_abi::fixed_size<4>>::value,
              "");

int main(int, char**) {
  return 0;
}
