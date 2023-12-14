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
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// constexpr index_type stride(rank_type i) const noexcept { return strides_[i]; }

// We intercept this inside layout_stride to give a consistent error message with
// layout_left and layout_right, technically the precondition is coming from the
// array access.

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  // value out of range
  {
    std::layout_stride::mapping<std::dextents<int, 3>> m(
        std::dextents<int, 3>(100, 100, 100), std::array<int, 3>{1, 100, 10000});

    TEST_LIBCPP_ASSERT_FAILURE(m.stride(4), "invalid rank index");
  }
  return 0;
}
