//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template <class It, class End>
// constexpr explicit(Extent != dynamic_extent) span(It first, End last);
// Requires: [first, last) shall be a valid range.
//   If Extent is not equal to dynamic_extent, then last - first shall be equal to Extent.
//

#include <span>
#include <iterator>


template<class T, size_t Extent>
std::span<T, Extent> createImplicitSpan(T* first, T* last) {
    return {first, last}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**) {
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(std::begin(arr), std::end(arr));

  return 0;
}
