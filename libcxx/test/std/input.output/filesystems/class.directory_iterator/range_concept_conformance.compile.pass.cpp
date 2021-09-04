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
// XFAIL: *

// directory_iterator, recursive_directory_iterator

#include "filesystem_include.h"

#include <concepts>
#include <ranges>



static_assert(std::same_as<std::ranges::iterator_t<fs::directory_iterator>, fs::directory_iterator>);
static_assert(std::ranges::common_range<fs::directory_iterator>);
static_assert(std::ranges::input_range<fs::directory_iterator>);
static_assert(!std::ranges::view<fs::directory_iterator>);
static_assert(!std::ranges::sized_range<fs::directory_iterator>);
static_assert(!std::ranges::borrowed_range<fs::directory_iterator>);
static_assert(!std::ranges::viewable_range<fs::directory_iterator>);

static_assert(std::same_as<std::ranges::iterator_t<fs::directory_iterator const>, fs::directory_iterator>);
static_assert(std::ranges::common_range<fs::directory_iterator const>);
static_assert(std::ranges::input_range<fs::directory_iterator const>);
static_assert(!std::ranges::view<fs::directory_iterator const>);
static_assert(!std::ranges::sized_range<fs::directory_iterator const>);
static_assert(!std::ranges::borrowed_range<fs::directory_iterator const>);
static_assert(!std::ranges::viewable_range<fs::directory_iterator const>);

static_assert(std::same_as<std::ranges::iterator_t<fs::recursive_directory_iterator>, fs::recursive_directory_iterator>);
static_assert(std::ranges::common_range<fs::recursive_directory_iterator>);
static_assert(std::ranges::input_range<fs::recursive_directory_iterator>);
static_assert(!std::ranges::view<fs::recursive_directory_iterator>);
static_assert(!std::ranges::sized_range<fs::recursive_directory_iterator>);
static_assert(!std::ranges::borrowed_range<fs::recursive_directory_iterator>);
static_assert(!std::ranges::viewable_range<fs::recursive_directory_iterator>);

static_assert(std::same_as<std::ranges::iterator_t<fs::recursive_directory_iterator const>, fs::recursive_directory_iterator>);
static_assert(std::ranges::common_range<fs::recursive_directory_iterator const>);
static_assert(std::ranges::input_range<fs::recursive_directory_iterator const>);
static_assert(!std::ranges::view<fs::recursive_directory_iterator const>);
static_assert(!std::ranges::sized_range<fs::recursive_directory_iterator const>);
static_assert(!std::ranges::borrowed_range<fs::recursive_directory_iterator const>);
static_assert(!std::ranges::viewable_range<fs::recursive_directory_iterator const>);
