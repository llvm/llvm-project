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
// template <class T, class U = typename T::value_type> struct memory_alignment;
// template <class T, class U = typename T::value_type>
//   inline constexpr std::size_t memory_alignment_v = memory_alignment<T, U>::value;

#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

int main(int, char**) {
  (void)ex::memory_alignment<ex::native_simd<bool>, bool>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::memory_alignment<int, int>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::memory_alignment<ex::native_simd_mask<int>, int>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::memory_alignment<ex::native_simd<int>, bool>::value;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}

  (void)ex::memory_alignment_v<ex::native_simd<bool>, bool>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::memory_alignment_v<int, int>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::memory_alignment_v<ex::native_simd_mask<int>, int>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}
  (void)ex::memory_alignment_v<ex::native_simd<int>, bool>;
  // expected-error-re@* {{no member named 'value' in {{.*}}}}

  return 0;
}
