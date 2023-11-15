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
// constexpr explicit observer_ptr(element_type* p) noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class ObjectT = T>
constexpr void test_element_type_ctor() {
  using Ptr = std::experimental::observer_ptr<T>;
  ObjectT obj;
  T* raw = &obj;
  Ptr ptr(raw);
  assert(ptr.get() == raw);
  static_assert(!std::is_convertible<T*, Ptr>::value);
  static_assert(std::is_nothrow_constructible<Ptr, T*>::value);
}

struct Bar {};

constexpr bool test() {
  test_element_type_ctor<Bar>();
  test_element_type_ctor<int>();
  test_element_type_ctor<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
