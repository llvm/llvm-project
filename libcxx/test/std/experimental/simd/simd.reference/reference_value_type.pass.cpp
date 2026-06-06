//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: target=powerpc{{.*}}le-unknown-linux-gnu

// <experimental/simd>
//
// [simd.reference]
// operator value_type() const noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdReferenceValueType {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> origin_simd([](T i) { return static_cast<T>(i); });
    for (size_t i = 0; i < origin_simd.size(); ++i) {
      static_assert(noexcept(T(origin_simd[i])));
      assert(T(origin_simd[i]) == static_cast<T>(i));
    }
  }
};

template <class T, std::size_t>
struct CheckMaskReferenceValueType {
  template <class SimdAbi>
  void operator()() {
    ex::simd_mask<T, SimdAbi> origin_simd_mask(true);
    for (size_t i = 0; i < origin_simd_mask.size(); ++i) {
      static_assert(noexcept(bool(origin_simd_mask[i])));
      assert(bool(origin_simd_mask[i]) == true);
    }
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdReferenceValueType>();
  test_all_simd_abi<CheckMaskReferenceValueType>();
  return 0;
}
