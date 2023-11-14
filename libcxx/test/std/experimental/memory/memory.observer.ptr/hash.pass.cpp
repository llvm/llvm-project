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
// template <class T> struct hash<std::experimental::observer_ptr<T>>;

#include <experimental/memory>
#include <cassert>

#include "poisoned_hash_helper.h"

template <class T, class Object = T>
void test_hash() {
  {
    using Ptr = std::experimental::observer_ptr<T>;
    Object obj;
    Ptr ptr(&obj);

    std::hash<std::experimental::observer_ptr<T>> f;
    std::size_t h = f(ptr);

    assert(h == std::hash<T*>()(&obj));
  }

  test_hash_enabled_for_type<std::experimental::observer_ptr<T>>();
}

struct Bar {};

void test() {
  test_hash<void, int>();
  test_hash<int>();
  test_hash<Bar>();
}

int main(int, char**) {
  // Note: This isn't constexpr friendly in the spec!
  test();

  return 0;
}
