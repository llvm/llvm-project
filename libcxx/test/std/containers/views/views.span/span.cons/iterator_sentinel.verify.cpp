//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// template<class It, class End>
//   constexpr explicit(extent != dynamic_extent) span(It first, End last);
//
// Preconditions:
//   - [first, last) is a valid range.
//
// Hardened preconditions:
//   If extent is not equal to dynamic_extent, then (last - first) == extent is true.

#include <cstddef>
#include <iterator>
#include <span>

template <class T, std::size_t Extent>
std::span<T, Extent> createImplicitSpan(T* first, T* last) {
  return {first, last}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

void f() {
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(std::begin(arr), std::end(arr));
}
