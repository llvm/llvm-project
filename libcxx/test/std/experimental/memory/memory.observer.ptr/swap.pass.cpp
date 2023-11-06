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
// constexpr void swap(observer_ptr& other) noexcept;
//
// template <class W>
// void swap(observer_ptr<W>& lhs, observer_ptr<W>& rhs) noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_swap() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj1, obj2;

  {
    Ptr ptr1(&obj1);
    Ptr ptr2(&obj2);

    assert(ptr1.get() == &obj1);
    assert(ptr2.get() == &obj2);

    ptr1.swap(ptr2);

    assert(ptr1.get() == &obj2);
    assert(ptr2.get() == &obj1);

    static_assert(noexcept(ptr1.swap(ptr2)));
    static_assert(std::is_same<decltype(ptr1.swap(ptr2)), void>::value);
  }

  {
    Ptr ptr1(&obj1);
    Ptr ptr2(&obj2);

    assert(ptr1.get() == &obj1);
    assert(ptr2.get() == &obj2);

    std::experimental::swap(ptr1, ptr2);

    assert(ptr1.get() == &obj2);
    assert(ptr2.get() == &obj1);

    static_assert(noexcept(std::experimental::swap(ptr1, ptr2)));
    static_assert(std::is_same<decltype(std::experimental::swap(ptr1, ptr2)), void>::value);
  }
}

struct Bar {};

constexpr bool test() {
  test_swap<Bar>();
  test_swap<int>();
  test_swap<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
