//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// template<class... Indices>
//   constexpr index_type operator()(Indices... i) const noexcept;
//
// Constraints:
//   - sizeof...(Indices) == extents_type::rank() is true,
//   - (is_convertible_v<Indices, index_type> && ...) is true, and
//   - (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
//
// Preconditions: extents_type::index-cast(i) is a multidimensional index in extents_ ([mdspan.overview]).

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  // value out of range
  {
    std::layout_right::template mapping<std::extents<unsigned char, 5>> m;
    TEST_LIBCPP_ASSERT_FAILURE(m(-1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(-130), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(1000), "layout_right::mapping: out of bounds indexing");
  }
  {
    std::layout_right::template mapping<std::extents<signed char, 5>> m;
    TEST_LIBCPP_ASSERT_FAILURE(m(-1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(-130), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(1000), "layout_right::mapping: out of bounds indexing");
  }
  {
    std::layout_right::template mapping<std::dextents<unsigned char, 1>> m(std::dextents<unsigned char, 1>(5));
    TEST_LIBCPP_ASSERT_FAILURE(m(-1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(-130), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(1000), "layout_right::mapping: out of bounds indexing");
  }
  {
    std::layout_right::template mapping<std::dextents<signed char, 1>> m(std::dextents<signed char, 1>(5));
    TEST_LIBCPP_ASSERT_FAILURE(m(-1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(-130), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(1000), "layout_right::mapping: out of bounds indexing");
  }
  {
    std::layout_right::template mapping<std::dextents<int, 3>> m(std::dextents<int, 3>(5, 7, 9));
    TEST_LIBCPP_ASSERT_FAILURE(m(-1, -1, -1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(-1, 0, 0), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(0, -1, 0), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(0, 0, -1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5, 3, 3), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(3, 7, 3), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(3, 3, 9), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5, 7, 9), "layout_right::mapping: out of bounds indexing");
  }
  {
    std::layout_right::template mapping<std::dextents<unsigned, 3>> m(std::dextents<int, 3>(5, 7, 9));
    TEST_LIBCPP_ASSERT_FAILURE(m(-1, -1, -1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(-1, 0, 0), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(0, -1, 0), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(0, 0, -1), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5, 3, 3), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(3, 7, 3), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(3, 3, 9), "layout_right::mapping: out of bounds indexing");
    TEST_LIBCPP_ASSERT_FAILURE(m(5, 7, 9), "layout_right::mapping: out of bounds indexing");
  }
  return 0;
}
