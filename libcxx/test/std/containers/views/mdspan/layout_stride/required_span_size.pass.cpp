//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// Let REQUIRED-SPAN-SIZE(e, strides) be:
//    - 1, if e.rank() == 0 is true,
//    - otherwise 0, if the size of the multidimensional index space e is 0,
//    - otherwise 1 plus the sum of products of (e.extent(r) - 1) and strides[r] for all r in the range [0, e.rank()).

// constexpr index_type required_span_size() const noexcept;
//
//   Returns: REQUIRED-SPAN-SIZE(extents(), strides_).

#include <mdspan>
#include <array>
#include <cassert>
#include <cstdint>
#include <span> // dynamic_extent

#include "test_macros.h"

template <class E>
constexpr void test_required_span_size(E e, std::array<int, E::rank()> strides, typename E::index_type expected_size) {
  using M = std::layout_stride::mapping<E>;
  const M m(e, strides);

  ASSERT_NOEXCEPT(m.required_span_size());
  assert(m.required_span_size() == expected_size);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;
  test_required_span_size(std::extents<int>(), std::array<int, 0>{}, 1);
  test_required_span_size(std::extents<unsigned, D>(0), std::array<int, 1>{5}, 0);
  test_required_span_size(std::extents<unsigned, D>(1), std::array<int, 1>{5}, 1);
  test_required_span_size(std::extents<unsigned, D>(7), std::array<int, 1>{5}, 31);
  test_required_span_size(std::extents<unsigned, 7>(), std::array<int, 1>{5}, 31);
  test_required_span_size(std::extents<unsigned, 7, 8>(), std::array<int, 2>{20, 2}, 135);
  test_required_span_size(
      std::extents<int64_t, D, 8, D, D>(7, 9, 10), std::array<int, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 5040);
  test_required_span_size(std::extents<int64_t, 1, 8, D, D>(9, 10), std::array<int, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 5034);
  test_required_span_size(std::extents<int64_t, 1, 0, D, D>(9, 10), std::array<int, 4>{1, 7, 7 * 8, 7 * 8 * 9}, 0);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
