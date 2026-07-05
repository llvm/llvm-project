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
// template <class W>
// std::experimental::observer_ptr<W> make_observer(W* p) noexcept;

#include <experimental/memory>
#include <cassert>
#include <type_traits>

template <class T, class Object = T>
void test_make_observer() {
  using Ptr = std::experimental::observer_ptr<T>;
  Object obj;
  T* raw = &obj;

  Ptr ptr = std::experimental::make_observer(raw);
  assert(ptr.get() == raw);
  static_assert(noexcept(std::experimental::make_observer(raw)));
  static_assert(std::is_same<decltype(std::experimental::make_observer(raw)), Ptr>::value);
}

struct Bar {};

void test() {
  test_make_observer<void, int>();
  test_make_observer<int>();
  test_make_observer<Bar>();
}

int main(int, char**) {
  // Note: this is not constexpr in the spec
  test();

  return 0;
}
