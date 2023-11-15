// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: c++experimental

// <experimental/memory>

// observer_ptr
//
// template <class W2>
// constexpr observer_ptr(observer_ptr<W2> other) noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class To, class From>
constexpr void test_converting_ctor() {
  using ToPtr   = std::experimental::observer_ptr<To>;
  using FromPtr = std::experimental::observer_ptr<From>;
  From obj;
  FromPtr from(&obj);
  ToPtr to = from;

  assert(from.get() == &obj);
  assert(to.get() == &obj);
#if TEST_STD_VER >= 20
  static_assert(std::is_nothrow_convertible<FromPtr, ToPtr>::value);
#endif
}

template <class To, class From>
constexpr void check_non_constructible() {
  using ToPtr   = std::experimental::observer_ptr<To>;
  using FromPtr = std::experimental::observer_ptr<From>;
  static_assert(!std::is_constructible<ToPtr, FromPtr>::value);
}

struct Bar {};
struct Base {};
struct Derived : Base {};

constexpr bool test() {
  test_converting_ctor<void, Bar>();
  test_converting_ctor<void, int>();
  test_converting_ctor<Base, Derived>();

  check_non_constructible<Derived, Base>();
  check_non_constructible<int, void>();
  check_non_constructible<Bar, void>();
  check_non_constructible<int, long>();
  check_non_constructible<long, int>();

  // Check const-ness
  check_non_constructible<Bar, Bar const>();
  test_converting_ctor<Bar const, Bar>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
