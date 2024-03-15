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

template <size_t I, typename C>
concept HasTupleElement = requires { std::tuple_element<I, C>{}; };

struct SomeObject {};

static_assert(!HasTupleElement<0, SomeObject>);
static_assert(!HasTupleElement<1, SomeObject>);
static_assert(!HasTupleElement<3, SomeObject>);

template <typename T>
void test() {
  using C = std::complex<T>;

  static_assert(HasTupleElement<0, C>);
  static_assert(HasTupleElement<1, C>);

  static_assert(std::same_as<typename std::tuple_element<0, C>::type, T>);
  static_assert(std::same_as<typename std::tuple_element<1, C>::type, T>);
}

void test() {
  test<float>();
  test<double>();
  test<long double>();
}
