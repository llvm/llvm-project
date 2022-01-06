//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// AppleClang 12.0.0 doesn't fully support ranges/concepts
// XFAIL: apple-clang-12.0.0

// <span>

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <span>
#include <cstddef>

template <class T, size_t extent>
std::span<T, extent> createImplicitSpan(T* ptr, size_t len) {
  return {ptr, len}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**) {
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(arr, 3);

  std::span<int> sp = {0, 0}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
  std::span<int, 2> sp2 = {0, 0}; // expected-error {{no matching constructor for initialization of 'std::span<int, 2>'}}
  std::span<const int> csp = {0, 0}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
  std::span<const int, 2> csp2 = {0, 0}; // expected-error {{no matching constructor for initialization of 'std::span<const int, 2>'}}

  return 0;
}
