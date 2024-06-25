//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <complex>

//   template<class T> struct tuple_size;

#include <cassert>
#include <complex>
#include <concepts>

template <typename C>
concept HasTupleSize = requires { std::tuple_size<C>{}; };

struct SomeObject {};

static_assert(!HasTupleSize<SomeObject>);

template <typename T>
void test() {
  using C = std::complex<T>;

  static_assert(HasTupleSize<C>);
  static_assert(std::same_as<typename std::tuple_size<C>::value_type, size_t>);
  static_assert(std::tuple_size<C>() == 2);
}

void test() {
  test<float>();
  test<double>();
  test<long double>();
}
