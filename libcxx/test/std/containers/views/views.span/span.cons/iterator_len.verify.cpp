//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// template<class It>
//   constexpr explicit(extent != dynamic_extent) span(It first, size_type count);
//
// Constraints: Let U be remove_reference_t<iter_reference_t<It>>.
//   - It satisfies contiguous_iterator.
//   - is_convertible_v<U(*)[], element_type(*)[]> is true.
//
// Preconditions:
//   - [first, first + count) is a valid range.
//
// Hardened preconditions:
//   - If extent is not equal to dynamic_extent, then count == extent is true.

#include <span>
#include <cstddef>

template <class T, std::size_t extent>
std::span<T, extent> createImplicitSpan(T* ptr, std::size_t len) {
  return {ptr, len}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

void test() {
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(arr, 3);

  // expected-error@+1 {{no matching constructor for initialization of 'std::span<int>'}}
  std::span<int> sp = {0, 0};
  // expected-error@+1 {{no matching constructor for initialization of 'std::span<int, 2>'}}
  std::span<int, 2> sp2 = {0, 0};
  // expected-error@+1 {{no matching constructor for initialization of 'std::span<const int>'}}
  std::span<const int> csp = {0, 0};
  // expected-error@+1 {{no matching constructor for initialization of 'std::span<const int, 2>'}}
  std::span<const int, 2> csp2 = {0, 0};
}
