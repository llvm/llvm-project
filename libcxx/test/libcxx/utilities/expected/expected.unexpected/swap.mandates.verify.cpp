//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

// Test the mandates
// constexpr void swap(unexpected& other) noexcept(is_nothrow_swappable_v<E>);
// Mandates: is_swappable_v<E> is true.

#include <expected>

struct Foo {};

void swap(Foo&, Foo&) = delete;

void test() {
  std::unexpected<Foo> f1{Foo{}};
  f1.swap(f1); // expected-note{{in instantiation of member function 'std::unexpected<Foo>::swap' requested here}}
  // expected-error-re@*:* {{static assertion failed {{.*}}unexpected::swap requires is_swappable_v<E> to be true}}
}
