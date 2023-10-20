//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// constexpr bool is_exhaustive() const noexcept;
//
// Returns:
//   - true if rank_ is 0.
//   - Otherwise, true if there is a permutation P of the integers in the range [0, rank_) such that
//     stride(p0) equals 1, and stride(pi) equals stride(pi_1) * extents().extent(pi_1) for i in the
//     range [1, rank_), where pi is the ith element of P.
//   - Otherwise, false.

#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

template <class E>
constexpr void
test_layout_mapping_stride(E ext, std::array<typename E::index_type, E::rank()> strides, bool exhaustive) {
  using M = std::layout_stride::template mapping<E>;
  M m(ext, strides);
  assert(m.is_exhaustive() == exhaustive);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;
  test_layout_mapping_stride(std::extents<int, 0>(), std::array<int, 1>{1}, true);
  test_layout_mapping_stride(std::extents<unsigned, D>(0), std::array<unsigned, 1>{3}, false);
  test_layout_mapping_stride(std::extents<int, 0, 3>(), std::array<int, 2>{6, 2}, true);
  test_layout_mapping_stride(std::extents<int, D, D>(3, 0), std::array<int, 2>{6, 2}, false);
  test_layout_mapping_stride(std::extents<int, D, D>(0, 0), std::array<int, 2>{6, 2}, false);
  test_layout_mapping_stride(
      std::extents<unsigned, D, D, D, D>(3, 3, 0, 3), std::array<unsigned, 4>{3, 1, 27, 9}, true);
  test_layout_mapping_stride(std::extents<int, D, D, D, D>(0, 3, 3, 3), std::array<int, 4>{3, 1, 27, 9}, false);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
