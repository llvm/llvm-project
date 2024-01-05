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
// constexpr element_type* get() const noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_get() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj;

  Ptr const ptr(&obj);
  assert(ptr.get() == &obj);

  static_assert(noexcept(ptr.get()));
  static_assert(std::is_same<decltype(ptr.get()), T*>::value);
}

struct Bar {};

constexpr bool test() {
  test_get<Bar>();
  test_get<int>();
  test_get<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
