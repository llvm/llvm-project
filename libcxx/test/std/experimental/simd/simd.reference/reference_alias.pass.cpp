//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Test the alias.
// simd::reference::value_type
// simd_mask::reference::value_type

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckRefAlias {
  template <class SimdAbi>
  void operator()() {
    static_assert(std::is_same_v<typename ex::simd<T, SimdAbi>::reference::value_type, T>);
    static_assert(std::is_same_v<typename ex::simd_mask<T, SimdAbi>::reference::value_type, bool>);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckRefAlias>();
  return 0;
}
