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
// reference operator[](size_t i);
// value_type operator[](size_t i) const;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdMaskReferenceSubscr {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> origin_simd_mask(true);
    for (size_t i = 0; i < origin_simd_mask.size(); ++i) {
      static_assert(noexcept(origin_simd_mask[i]));
      static_assert(std::is_same_v<typename ex::simd_mask<T, SimdAbi>::reference, decltype(origin_simd_mask[i])>);
      assert(origin_simd_mask[i] == true);
    }
  }
};

template <class T, std::size_t>
struct CheckSimdMaskValueTypeSubscr {
  template <class SimdAbi>
  void operator()() {
    const ex::simd_mask<T, SimdAbi> origin_simd_mask(true);
    for (size_t i = 0; i < origin_simd_mask.size(); ++i) {
      static_assert(noexcept(origin_simd_mask[i]));
      static_assert(std::is_same_v<bool, decltype(origin_simd_mask[i])>);
      assert(origin_simd_mask[i] == true);
    }
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdMaskReferenceSubscr>();
  test_all_simd_abi<CheckSimdMaskValueTypeSubscr>();
  return 0;
}
