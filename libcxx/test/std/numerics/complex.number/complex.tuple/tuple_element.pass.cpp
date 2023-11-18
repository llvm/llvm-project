//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

// template <size_t I, class T>
// struct tuple_element<I, complex<T>>;

#include <complex>

template <typename T>
void test() {
  static_assert(std::is_same_v<typename std::tuple_element<0, std::complex<T>>::type, T>);
  static_assert(std::is_same_v<typename std::tuple_element<1, std::complex<T>>::type, T>);
}

int main() {
  test<float>();
  test<double>();
  test<long double>();
}
