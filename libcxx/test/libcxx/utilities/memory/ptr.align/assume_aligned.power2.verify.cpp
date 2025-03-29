//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

// This test checks that we static_assert inside std::assume_aligned<N>(p)
// when N is not a power of two. However, Clang will already emit an error
// in its own __builtin_assume_aligned, so we ignore that additional error
// for the purpose of this test. We also ignore the additional warning about
// remainder by 0 being undefined.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error -Xclang -verify-ignore-unexpected=warning

#include <memory>

void f() {
  int *p = nullptr;
  (void)std::assume_aligned<0>(p);  // expected-error@*:* {{std::assume_aligned<N>(p) requires N to be a power of two}}
  (void)std::assume_aligned<3>(p);  // expected-error@*:* {{std::assume_aligned<N>(p) requires N to be a power of two}}
  (void)std::assume_aligned<5>(p);  // expected-error@*:* {{std::assume_aligned<N>(p) requires N to be a power of two}}
  (void)std::assume_aligned<33>(p); // expected-error@*:* {{std::assume_aligned<N>(p) requires N to be a power of two}}
}
