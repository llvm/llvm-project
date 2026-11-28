//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// pointer address(reference x) const;
// const_pointer address(const_reference x) const;

// In C++20, parts of std::allocator<T> have been removed.
// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <memory>

template <class T>
void test_address(const std::allocator<T>& a, T& x, const T& cx) {
  (void)a.address(x);  // expected-error 2 {{no member}}
  (void)a.address(cx); // expected-error 2 {{no member}}
}

struct A {
  void operator&() const {}
};

void test(const std::allocator<int>& ai, int& i, const std::allocator<A>& aa, A& x) {
  test_address(ai, i, i);
  test_address(aa, x, x);
}
