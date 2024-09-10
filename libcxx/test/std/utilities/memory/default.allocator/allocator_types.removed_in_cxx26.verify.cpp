//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <memory>

// Check that the following nested types are removed in C++26:

// template <class T>
// class allocator
// {
// ...
//     typedef true_type is_always_equal; // Deprecated in C++23, removed in C++26
// ...
// };

#include <memory>

template <typename T>
void check() {
  using IAE = typename std::allocator<T>::is_always_equal; // expected-error 3 {{no type named 'is_always_equal'}}
}

void test() {
  check<char>();
  check<int>();
  check<void>();
}
