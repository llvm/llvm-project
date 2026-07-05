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
// constexpr explicit operator bool() const noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_operator_bool() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj;

  {
    Ptr const ptr(&obj);
    bool b = static_cast<bool>(ptr);
    assert(b);

    static_assert(noexcept(static_cast<bool>(ptr)));
  }

  {
    Ptr const ptr(nullptr);
    bool b = static_cast<bool>(ptr);
    assert(!b);
  }

  static_assert(!std::is_convertible<Ptr const, bool>::value);
  static_assert(std::is_constructible<bool, Ptr const>::value);
}

struct Bar {};

constexpr bool test() {
  test_operator_bool<Bar>();
  test_operator_bool<int>();
  test_operator_bool<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
