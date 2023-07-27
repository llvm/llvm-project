//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-has-hardened-mode && !libcpp-has-debug-mode
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// constexpr mapping(const extents_type& e) noexcept;
//
// Preconditions: The size of the multidimensional index space e is representable as a value of type index_type ([basic.fundamental]).
//
// Effects: Direct-non-list-initializes extents_ with e.

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;

  // value out of range
  {
    // the extents are representable but the product is not, so we can't use it for layout_right
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_right::template mapping<std::extents<char, D, 5>> m(std::extents<char, D, 5>(100)); }()),
        "layout_right::mapping extents ctor: product of extents must be representable as index_type.");
  }
  return 0;
}
