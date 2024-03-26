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
// constexpr observer_ptr(nullptr_t) noexcept;

#include <experimental/memory>
#include <cassert>
#include <cstddef>
#include <type_traits>

template <class T>
constexpr void test_nullptr_ctor() {
  using Ptr = std::experimental::observer_ptr<T>;
  Ptr ptr   = nullptr;
  assert(ptr.get() == nullptr);
  static_assert(std::is_nothrow_constructible<Ptr, std::nullptr_t>::value);
}

struct Foo;
struct Bar {
  Bar(int) {}
};

constexpr bool test() {
  test_nullptr_ctor<Foo>();
  test_nullptr_ctor<Bar>();
  test_nullptr_ctor<int>();
  test_nullptr_ctor<void>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
