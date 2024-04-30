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
// constexpr element_type* release() noexcept;

#include <experimental/memory>
#include <type_traits>
#include <cassert>

template <class T, class Object = T>
constexpr void test_release() {
  Object obj;
  using Ptr = std::experimental::observer_ptr<T>;
  Ptr ptr(&obj);
  assert(ptr.get() == &obj);

  decltype(auto) r = ptr.release();
  assert(r == &obj);
  assert(ptr.get() == nullptr);

  static_assert(std::is_same<decltype(r), typename Ptr::element_type*>::value);
  static_assert(noexcept(ptr.release()));
}

struct Bar {};

constexpr bool test() {
  test_release<Bar>();
  test_release<int>();
  test_release<void, int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
