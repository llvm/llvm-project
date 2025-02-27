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

// template<class OtherIndexType>
//   constexpr mapping(const extents_type& e, const array<OtherIndexType, rank_>& s) noexcept;
//
// Constraints:
//   - is_convertible_v<const OtherIndexType&, index_type> is true, and
//   - is_nothrow_constructible_v<index_type, const OtherIndexType&> is true.
//
// Preconditions:
//   - s[i] > 0 is true for all i in the range [0, rank_).
//   - REQUIRED-SPAN-SIZE(e, s) is representable as a value of type index_type ([basic.fundamental]).
//   - If rank_ is greater than 0, then there exists a permutation P of the integers in the range [0, rank_),
//     such that s[pi] >= s[pi_1] * e.extent(pi_1) is true for all i in the range [1, rank_), where
//     pi is the ith element of P.
//     [Note 1: For layout_stride, this condition is necessary and sufficient for is_unique() to be true. end note]
//
// Effects: Direct-non-list-initializes extents_ with e, and for all d in the range [0, rank_), direct-non-list-initializes strides_[d] with as_const(s[d]).

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;

  // the extents are representable but the product with strides is not, so we can't use it for layout_stride
  TEST_LIBCPP_ASSERT_FAILURE(
      ([=] {
        std::layout_stride::mapping<std::extents<signed char, D, 5>> m(
            std::extents<signed char, D, 5>(20), std::array<int, 2>{20, 1});
      }()),
      "layout_stride::mapping ctor: required span size is not representable as index_type.");

  // check that if we first overflow in strides conversion we also fail
  static_assert(static_cast<unsigned char>(257u) == 1);
  TEST_LIBCPP_ASSERT_FAILURE(
      ([=] {
        std::layout_stride::mapping<std::extents<unsigned char, D, 5>> m(
            std::extents<unsigned char, D, 5>(20), std::array<unsigned, 2>{257, 1});
      }()),
      "layout_stride::mapping ctor: required span size is not representable as index_type.");

  // negative strides are not allowed, check with unsigned index_type so we make sure we catch that
  TEST_LIBCPP_ASSERT_FAILURE(
      ([=] {
        std::layout_stride::mapping<std::extents<unsigned, D, 5>> m(
            std::extents<unsigned, D, 5>(20), std::array<int, 2>{20, -1});
      }()),
      "layout_stride::mapping ctor: all strides must be greater than 0");
  // zero strides are not allowed, check with unsigned index_type so we make sure we catch that
  TEST_LIBCPP_ASSERT_FAILURE(
      ([=] {
        std::layout_stride::mapping<std::extents<unsigned, D, 5>> m(
            std::extents<unsigned, D, 5>(20), std::array<unsigned, 2>{20, 0});
      }()),
      "layout_stride::mapping ctor: all strides must be greater than 0");
  return 0;
}
