//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

// template <class T>
// struct tuple_size<complex<T>>;

#include <complex>

template <typename T>
void test() {
  static_assert(std::tuple_size<std::complex<T>>::value == 2);
}

int main() {
  test<float>();
  test<double>();
  test<long double>();
}
