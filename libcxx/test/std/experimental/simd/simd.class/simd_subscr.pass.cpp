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
// [simd.class]
// reference operator[](size_t i);
// value_type operator[](size_t i) const;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdReferenceSubscr {
  template <class SimdAbi>
  void operator()() {
    ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    for (size_t i = 0; i < origin_simd.size(); ++i) {
      static_assert(noexcept(origin_simd[i]));
      static_assert(std::is_same_v<typename ex::simd<T, SimdAbi>::reference, decltype(origin_simd[i])>);
      assert(origin_simd[i] == static_cast<T>(i));
    }
  }
};

template <class T, std::size_t>
struct CheckSimdValueTypeSubscr {
  template <class SimdAbi>
  void operator()() {
    const ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
    for (size_t i = 0; i < origin_simd.size(); ++i) {
      static_assert(noexcept(origin_simd[i]));
      static_assert(std::is_same_v<T, decltype(origin_simd[i])>);
      assert(origin_simd[i] == static_cast<T>(i));
    }
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdReferenceSubscr>();
  test_all_simd_abi<CheckSimdValueTypeSubscr>();
  return 0;
}
