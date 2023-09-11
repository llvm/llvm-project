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

// constexpr size_type size() const noexcept;
//
// Preconditions: The size of the multidimensional index space extents() is representable as a value of type size_type ([basic.fundamental]).
//
// Returns: extents().fwd-prod-of-extents(rank()).

#include <array>
#include <cassert>
#include <mdspan>

#include "check_assertion.h"
#include "../../../../../std/containers/views/mdspan/mdspan/CustomTestLayouts.h"

// We use a funky mapping in this test where required_span_size is much smaller than the size of the index space
int main(int, char**) {
  std::array<float, 10> data;
  // make sure we are not failing because of using index_type instead of size_type
  {
    typename layout_wrapping_integral<4>::template mapping<std::dextents<char, 2>> map(
      std::dextents<char, 2>(100, 2), not_extents_constructible_tag());
    std::mdspan<float, std::dextents<char, 2>, layout_wrapping_integral<4>> mds(data.data(), map);
    assert(map.required_span_size() == char(8));
    assert((static_cast<unsigned char>(200) == mds.size()));
  }
  {
    typename layout_wrapping_integral<4>::template mapping<std::dextents<char, 2>> map(
      std::dextents<char, 2>(100, 3), not_extents_constructible_tag());
    std::mdspan<float, std::dextents<char, 2>, layout_wrapping_integral<4>> mds(data.data(), map);
    // sanity check
    assert(map.required_span_size() == char(12));
    // 100 x 3 exceeds 256
    {
      TEST_LIBCPP_ASSERT_FAILURE(([=] { mds.size(); }()), "mdspan: size() is not representable as size_type");
    }
  }
  return 0;
}
