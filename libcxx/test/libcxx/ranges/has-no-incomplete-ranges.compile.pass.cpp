//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-filesystem-library
// REQUIRES: libcpp-has-no-incomplete-ranges

// Test that _LIBCPP_HAS_NO_INCOMPLETE_RANGES disables the std::ranges namespace.

#include <algorithm>
#include <concepts>
#include <filesystem>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace ranges {
  int output_range;
  int data;
  int size;
  int prev;
  int next;
  int distance;
  int take_view;
  int drop_view;
  int transform_view;
  int filter_view;
  int join_view;
  int views; // this entire namespace should be absent
}
_LIBCPP_END_NAMESPACE_STD
