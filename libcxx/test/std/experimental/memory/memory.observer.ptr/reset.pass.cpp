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
// constexpr void reset(element_type* p = nullptr) noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_reset() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj1, obj2;

  {
    Ptr ptr(&obj1);
    assert(ptr.get() == &obj1);
    ptr.reset(&obj2);
    assert(ptr.get() == &obj2);
    static_assert(noexcept(ptr.reset(&obj2)));
    static_assert(std::is_same<decltype(ptr.reset(&obj2)), void>::value);
  }
  {
    Ptr ptr(&obj1);
    assert(ptr.get() == &obj1);
    ptr.reset();
    assert(ptr.get() == nullptr);
    static_assert(noexcept(ptr.reset()));
    static_assert(std::is_same<decltype(ptr.reset()), void>::value);
  }
}

struct Bar {};

constexpr bool test() {
  test_reset<Bar>();
  test_reset<int>();
  test_reset<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
