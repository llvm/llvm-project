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
// constexpr explicit operator element_type*() const noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_convertibility() {
  Object obj;
  std::experimental::observer_ptr<T> ptr(&obj);

  T* raw = static_cast<T*>(ptr);
  assert(raw == &obj);
  static_assert(!std::is_convertible<std::experimental::observer_ptr<T>, T*>::value, "");
  static_assert(std::is_constructible<T*, std::experimental::observer_ptr<T>>::value, "");
}

struct Incomplete;

struct Bar {};

constexpr bool test() {
  test_convertibility<void, int>();
  test_convertibility<bool>();
  test_convertibility<int>();
  test_convertibility<Bar>();

  {
    std::experimental::observer_ptr<Incomplete> ptr = nullptr;
    Incomplete* raw                                 = static_cast<Incomplete*>(ptr);
    assert(raw == nullptr);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
