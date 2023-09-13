//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Test the alias.
// template <class T, class Abi> class simd {
// public:
//   using value_type = T;
//   using mask_type = simd_mask<T, Abi>;
//   using abi_type = Abi;
// };

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdAlias {
  template <class SimdAbi>
  void operator()() {
    static_assert(std::is_same_v<typename ex::simd<T, SimdAbi>::value_type, T>);
    static_assert(std::is_same_v<typename ex::simd<T, SimdAbi>::mask_type, ex::simd_mask<T, SimdAbi>>);
    static_assert(std::is_same_v<typename ex::simd<T, SimdAbi>::abi_type, SimdAbi>);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdAlias>();
  return 0;
}
