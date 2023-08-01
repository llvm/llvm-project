//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=unchecked
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// template<class OtherElementType, class OtherExtents,
//         class OtherLayoutPolicy, class OtherAccessor>
//  constexpr explicit(see below)
//    mdspan(const mdspan<OtherElementType, OtherExtents,
//                        OtherLayoutPolicy, OtherAccessor>& other);
//
// Constraints:
//   - is_constructible_v<mapping_type, const OtherLayoutPolicy::template mapping<OtherExtents>&> is true, and
//   - is_constructible_v<accessor_type, const OtherAccessor&> is true.
// Mandates:
//   - is_constructible_v<data_handle_type, const OtherAccessor::data_handle_type&> is
//   - is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions:
//   - For each rank index r of extents_type, static_extent(r) == dynamic_extent || static_extent(r) == other.extent(r) is true.
//   - [0, map_.required_span_size()) is an accessible range of ptr_ and acc_ for values of ptr_, map_, and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with other.ptr_,
//   - direct-non-list-initializes map_ with other.map_, and
//   - direct-non-list-initializes acc_ with other.acc_.
//
// Remarks: The expression inside explicit is equivalent to:
//   !is_convertible_v<const OtherLayoutPolicy::template mapping<OtherExtents>&, mapping_type>
//   || !is_convertible_v<const OtherAccessor&, accessor_type>

#include <array>
#include <cassert>
#include <mdspan>

#include "check_assertion.h"
#include "../../../../../std/containers/views/mdspan/mdspan/CustomTestLayouts.h"

// We use a funky mapping in this test that doesn't check the dynamic/static extents mismatch itself
int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;
  std::array<float, 10> data;
  typename layout_wrapping_integral<4>::template mapping<std::dextents<int, 2>> src_map(
      std::dextents<int, 2>(5, 2), not_extents_constructible_tag());
  std::mdspan<float, std::dextents<int, 2>, layout_wrapping_integral<4>> arg(data.data(), src_map);

  // working case
  {
    [[maybe_unused]] std::mdspan<float, std::extents<size_t, D, 2>, layout_wrapping_integral<4>> m(arg); // should work
  }
  // mismatch of static extent
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::mdspan<float, std::extents<size_t, D, 3>, layout_wrapping_integral<4>> m(arg); }()),
        "mdspan: conversion mismatch of source dynamic extents with static extents");
  }
  return 0;
}
