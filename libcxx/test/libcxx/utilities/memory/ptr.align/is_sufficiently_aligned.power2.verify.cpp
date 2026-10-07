//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// #include <memory>

// template<size_t Alignment, class T>
//   bool is_sufficiently_aligned(T* ptr);

#include <memory>

void f() {
  int* p = nullptr;
  (void)std::is_sufficiently_aligned<0>(
      p); // expected-error@*:* {{std::is_sufficiently_aligned<Alignment>(p) requires Alignment to be a power of two}}
  (void)std::is_sufficiently_aligned<3>(
      p); // expected-error@*:* {{std::is_sufficiently_aligned<Alignment>(p) requires Alignment to be a power of two}}
  (void)std::is_sufficiently_aligned<5>(
      p); // expected-error@*:* {{std::is_sufficiently_aligned<Alignment>(p) requires Alignment to be a power of two}}
  (void)std::is_sufficiently_aligned<33>(
      p); // expected-error@*:* {{std::is_sufficiently_aligned<Alignment>(p) requires Alignment to be a power of two}}
}
