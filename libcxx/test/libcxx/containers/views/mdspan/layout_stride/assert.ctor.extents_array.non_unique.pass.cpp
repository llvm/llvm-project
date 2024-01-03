//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing

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

  // overlapping strides
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] {
          std::layout_stride::mapping<std::extents<unsigned, D, 5, 7>> m(
              std::extents<unsigned, D, 5, 7>(20), std::array<unsigned, 3>{4, 1, 200});
        }()),
        "layout_stride::mapping ctor: the provided extents and strides lead to a non-unique mapping");
  }
  // equal strides
  {
    // should work because one of the equal strides is associated with an extent of 1
    [[maybe_unused]] std::layout_stride::mapping<std::extents<unsigned, D, 5, 1>> m1(
        std::extents<unsigned, D, 5, 1>(2), std::array<unsigned, 3>{5, 1, 5});
    [[maybe_unused]] std::layout_stride::mapping<std::extents<unsigned, D, 5, 2>> m2(
        std::extents<unsigned, D, 5, 2>(1), std::array<unsigned, 3>{5, 1, 5});

    // will fail because neither of the equal strides is associated with an extent of 1
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] {
          std::layout_stride::mapping<std::extents<unsigned, D, 5, 2>> m3(
              std::extents<unsigned, D, 5, 2>(2), std::array<unsigned, 3>{5, 1, 5});
        }()),
        "layout_stride::mapping ctor: the provided extents and strides lead to a non-unique mapping");
  }
  return 0;
}
