//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

//   template<size_t I, class T> struct tuple_element;

#include <cassert>
#include <complex>
#include <concepts>

template <typename T>
void test() {
  using C = std::complex<T>;

  // expected-error-re@*:* 3 {{static assertion failed {{.*}}Index value is out of range.}}
  [[maybe_unused]] std::tuple_element<3, C> te{};
}

void test() {
  test<float>();
  test<double>();
  test<long double>();
}
