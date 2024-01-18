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
// constexpr std::add_lvalue_reference_t<element_type> operator*() const;
// constexpr element_type* operator->() const noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_deref() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj;

  {
    Ptr const ptr(&obj);
    T& r = *ptr;
    assert(&r == &obj);
  }
  {
    Ptr const ptr(&obj);
    T* r = ptr.operator->();
    assert(r == &obj);
    static_assert(noexcept(ptr.operator->()));
  }
}

struct Bar {};
struct Foo {
  int member = 42;
};

constexpr bool test() {
  test_deref<Bar>();
  test_deref<int>();

  {
    Foo foo;
    std::experimental::observer_ptr<Foo> ptr(&foo);
    assert(&ptr->member == &foo.member);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
