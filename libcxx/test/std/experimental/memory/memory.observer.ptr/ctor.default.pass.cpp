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
// constexpr observer_ptr() noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T>
constexpr void test_default_ctor() {
  using Ptr = std::experimental::observer_ptr<T>;
  Ptr ptr;
  assert(ptr.get() == nullptr);
  static_assert(std::is_nothrow_default_constructible<Ptr>::value);
}

struct Foo;
struct Bar {
  Bar(int) {}
};

constexpr bool test() {
  test_default_ctor<Foo>();
  test_default_ctor<Bar>();
  test_default_ctor<int>();
  test_default_ctor<void>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
