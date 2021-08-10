//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges
// XFAIL: *

// directory_iterator, recursive_directory_iterator

#include "filesystem_include.h"

#include <concepts>
#include <ranges>

namespace stdr = std::ranges;

static_assert(std::same_as<stdr::iterator_t<fs::directory_iterator>, fs::directory_iterator>);
static_assert(stdr::common_range<fs::directory_iterator>);
static_assert(stdr::input_range<fs::directory_iterator>);
static_assert(!stdr::view<fs::directory_iterator>);
static_assert(!stdr::sized_range<fs::directory_iterator>);
static_assert(!stdr::borrowed_range<fs::directory_iterator>);
static_assert(!stdr::viewable_range<fs::directory_iterator>);

static_assert(std::same_as<stdr::iterator_t<fs::directory_iterator const>, fs::directory_iterator>);
static_assert(stdr::common_range<fs::directory_iterator const>);
static_assert(stdr::input_range<fs::directory_iterator const>);
static_assert(!stdr::view<fs::directory_iterator const>);
static_assert(!stdr::sized_range<fs::directory_iterator const>);
static_assert(!stdr::borrowed_range<fs::directory_iterator const>);
static_assert(!stdr::viewable_range<fs::directory_iterator const>);

static_assert(std::same_as<stdr::iterator_t<fs::recursive_directory_iterator>, fs::recursive_directory_iterator>);
static_assert(stdr::common_range<fs::recursive_directory_iterator>);
static_assert(stdr::input_range<fs::recursive_directory_iterator>);
static_assert(!stdr::view<fs::recursive_directory_iterator>);
static_assert(!stdr::sized_range<fs::recursive_directory_iterator>);
static_assert(!stdr::borrowed_range<fs::recursive_directory_iterator>);
static_assert(!stdr::viewable_range<fs::recursive_directory_iterator>);

static_assert(std::same_as<stdr::iterator_t<fs::recursive_directory_iterator const>, fs::recursive_directory_iterator>);
static_assert(stdr::common_range<fs::recursive_directory_iterator const>);
static_assert(stdr::input_range<fs::recursive_directory_iterator const>);
static_assert(!stdr::view<fs::recursive_directory_iterator const>);
static_assert(!stdr::sized_range<fs::recursive_directory_iterator const>);
static_assert(!stdr::borrowed_range<fs::recursive_directory_iterator const>);
static_assert(!stdr::viewable_range<fs::recursive_directory_iterator const>);
