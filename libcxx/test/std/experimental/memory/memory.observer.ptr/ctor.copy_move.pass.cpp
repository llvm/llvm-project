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
// observer_ptr(const observer_ptr& other) = default;
// observer_ptr(observer_ptr&& other) = default;

#include <experimental/memory>
#include <cassert>
#include <type_traits>
#include <utility>

template <class T, class Object = T>
constexpr void test_copy_move() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj;
  {
    Ptr ptr(&obj);
    Ptr copy = ptr;
    assert(ptr.get() == &obj);
    assert(copy.get() == &obj);
    static_assert(std::is_nothrow_copy_constructible<Ptr>::value);
  }
  {
    Ptr ptr(&obj);
    Ptr copy = std::move(ptr);
    assert(ptr.get() == &obj);
    assert(copy.get() == &obj);
    static_assert(std::is_nothrow_move_constructible<Ptr>::value);
  }
}

struct Bar {};

constexpr bool test() {
  test_copy_move<int>();
  test_copy_move<Bar>();
  test_copy_move<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}