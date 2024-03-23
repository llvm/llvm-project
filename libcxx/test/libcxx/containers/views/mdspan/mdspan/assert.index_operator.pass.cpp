//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <mdspan>

// template<class... OtherIndexTypes>
//   constexpr reference operator[](OtherIndexTypes... indices) const;
// Constraints:
//   - (is_convertible_v<OtherIndexTypes, index_type> && ...) is true,
//   - (is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) is true, and
//   - sizeof...(OtherIndexTypes) == rank() is true.
//
// Let I be extents_type::index-cast(std::move(indices)).
//
// Preconditions: I is a multidimensional index in extents().
//   Note 1: This implies that map_(I) < map_.required_span_size() is true.
//
// Effects: Equivalent to:
//   return acc_.access(ptr_, map_(static_cast<index_type>(std::move(indices))...));

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  float data[1024];
  // value out of range
  {
    std::mdspan m(data, std::extents<unsigned char, 5>());
    TEST_LIBCPP_ASSERT_FAILURE(m[-1], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[-130], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[5], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[1000], "mdspan: operator[] out of bounds access");
  }
  {
    std::mdspan m(data, std::extents<signed char, 5>());
    TEST_LIBCPP_ASSERT_FAILURE(m[-1], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[-130], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[5], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[1000], "mdspan: operator[] out of bounds access");
  }
  {
    std::mdspan m(data, std::dextents<unsigned char, 1>(5));
    TEST_LIBCPP_ASSERT_FAILURE(m[-1], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[-130], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[5], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[1000], "mdspan: operator[] out of bounds access");
  }
  {
    std::mdspan m(data, std::dextents<signed char, 1>(5));
    TEST_LIBCPP_ASSERT_FAILURE(m[-1], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[-130], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[5], "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE(m[1000], "mdspan: operator[] out of bounds access");
  }
  {
    std::mdspan m(data, std::dextents<int, 3>(5, 7, 9));
    TEST_LIBCPP_ASSERT_FAILURE((m[-1, -1, -1]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[-1, 0, 0]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[0, -1, 0]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[0, 0, -1]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[5, 3, 3]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[3, 7, 3]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[3, 3, 9]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[5, 7, 9]), "mdspan: operator[] out of bounds access");
  }
  {
    std::mdspan m(data, std::dextents<unsigned, 3>(5, 7, 9));
    TEST_LIBCPP_ASSERT_FAILURE((m[-1, -1, -1]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[-1, 0, 0]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[0, -1, 0]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[0, 0, -1]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[5, 3, 3]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[3, 7, 3]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[3, 3, 9]), "mdspan: operator[] out of bounds access");
    TEST_LIBCPP_ASSERT_FAILURE((m[5, 7, 9]), "mdspan: operator[] out of bounds access");
  }
  return 0;
}
