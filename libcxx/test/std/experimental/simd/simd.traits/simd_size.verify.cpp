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
//template <class T, class Abi = simd_abi::compatible<T>> struct simd_size;
//template <class T, class Abi = simd_abi::compatible<T>>
//inline constexpr std::size_t ex::simd_size_v = ex::simd_size<T, Abi>::value;

#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

int main(int, char**) {
  (void)ex::simd_size<bool, ex::simd_abi::compatible<bool>>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::simd_size<ex::native_simd<int>, ex::simd_abi::native<int>>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::simd_size<int, int>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::simd_size<int, ex::native_simd<int>>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}

  (void)ex::simd_size_v<bool, ex::simd_abi::compatible<bool>>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::simd_size_v<ex::native_simd<int>, ex::simd_abi::native<int>>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::simd_size_v<int, int>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::simd_size_v<int, ex::native_simd<int>>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}

  return 0;
}
