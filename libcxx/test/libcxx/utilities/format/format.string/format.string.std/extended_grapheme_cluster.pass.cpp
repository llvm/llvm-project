//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// Tests the implementation of the extended grapheme cluster boundaries per
// https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundary_Rules
//
// The tests are based on the test data provided by Unicode
// https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/GraphemeBreakTest.txt

#include <cassert>
#include <format>
#include <functional>
#include <numeric>

#include "extended_grapheme_cluster.h"

// Validates whether the number of code points in our "database" matches with
// the number in the Unicode. The assumption is when the number of items per
// property matches the code points themselves also match.
namespace {
namespace cluster = std::__extended_grapheme_custer_property_boundary;
constexpr int count_entries(cluster::__property property) {
  return std::transform_reduce(
      std::begin(cluster::__entries), std::end(cluster::__entries), 0, std::plus{}, [property](auto entry) {
        if (static_cast<cluster::__property>(entry & 0xf) != property)
          return 0;

        return 1 + static_cast<int>((entry >> 4) & 0x7f);
      });
}

static_assert(count_entries(cluster::__property::__Prepend) == 27);
static_assert(count_entries(cluster::__property::__CR) == 1);
static_assert(count_entries(cluster::__property::__LF) == 1);
static_assert(count_entries(cluster::__property::__Control) == 3893);
static_assert(count_entries(cluster::__property::__Extend) == 2130);
static_assert(count_entries(cluster::__property::__Regional_Indicator) == 26);
static_assert(count_entries(cluster::__property::__SpacingMark) == 395);
static_assert(count_entries(cluster::__property::__L) == 125);
static_assert(count_entries(cluster::__property::__V) == 95);
static_assert(count_entries(cluster::__property::__T) == 137);
static_assert(count_entries(cluster::__property::__LV) == 399);
static_assert(count_entries(cluster::__property::__LVT) == 10773);
static_assert(count_entries(cluster::__property::__ZWJ) == 1);
static_assert(count_entries(cluster::__property::__Extended_Pictographic) == 3537);

} // namespace

template <class Data>
constexpr void test(const Data& data) {
  for (const auto& d : data) {
    assert(d.code_points.size() == d.breaks.size());

    std::__unicode::__extended_grapheme_cluster_view view{d.input.begin(), d.input.end()};
    for (std::size_t i = 0; i < d.breaks.size(); ++i) {
      auto r = view.__consume();
      assert(r.__code_point_ == d.code_points[i]);
      assert(r.__last_ == d.input.begin() + d.breaks[i]);
    }
  }
}

constexpr bool test() {
  test(data_utf8);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  if constexpr (sizeof(wchar_t) == 2)
    test(data_utf16);
  else
    test(data_utf32);
#endif

  return true;
}

int main(int, char**) {
  test();
  // static_assert(test());

  return 0;
}
