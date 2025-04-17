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
#include <cassert>

#include "test_macros.h"

template <class T>
void test_address() {
  T* tp        = new T();
  const T* ctp = tp;
  const std::allocator<T> a;
  assert(a.address(*tp) == tp);  // expected-error 2 {{no member}}
  assert(a.address(*ctp) == tp); // expected-error 2 {{no member}}
  delete tp;
}

struct A {
  void operator&() const {}
};

int main(int, char**) {
  test_address<int>();
  test_address<A>();

  return 0;
}
