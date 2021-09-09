//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// class std::ranges::subrange;

#include <ranges>

#include "test_iterators.h"

namespace ranges = std::ranges;

static_assert(ranges::borrowed_range<ranges::subrange<int*>>);
static_assert(ranges::borrowed_range<ranges::subrange<int*, int const*>>);
static_assert(ranges::borrowed_range<ranges::subrange<int*, sentinel_wrapper<int*>, ranges::subrange_kind::unsized>>);
