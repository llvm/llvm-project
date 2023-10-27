//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template<size_t Count>
//  constexpr span<element_type, Count> first() const;
//
// Mandates: Count <= Extent is true.

#include <span>
#include <cstddef>

void f() {
  int array[] = {1, 2, 3, 4};
  std::span<const int, 4> sp(array);

  //  Count too large
  [[maybe_unused]] auto s1 = sp.first<5>(); // expected-error@span:* {{span<T, N>::first<Count>(): Count out of range}}

  //  Count numeric_limits
  [[maybe_unused]] auto s2 = sp.first<std::size_t(-1)>(); // expected-error@span:* {{span<T, N>::first<Count>(): Count out of range}}
}
