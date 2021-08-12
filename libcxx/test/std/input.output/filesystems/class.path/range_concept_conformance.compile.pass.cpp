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

// path

#include "filesystem_include.h"

#include <concepts>
#include <ranges>

namespace stdr = std::ranges;

static_assert(std::same_as<stdr::iterator_t<fs::path>, fs::path::iterator>);
static_assert(stdr::common_range<fs::path>);
static_assert(stdr::bidirectional_range<fs::path>);
static_assert(!stdr::view<fs::path>);
static_assert(!stdr::random_access_range<fs::path>);
static_assert(!stdr::sized_range<fs::path>);
static_assert(!stdr::borrowed_range<fs::path>);
static_assert(!stdr::viewable_range<fs::path>);

static_assert(std::same_as<stdr::iterator_t<fs::path const>, fs::path::const_iterator>);
static_assert(stdr::common_range<fs::path const>);
static_assert(stdr::bidirectional_range<fs::path const>);
static_assert(!stdr::view<fs::path const>);
static_assert(!stdr::random_access_range<fs::path const>);
static_assert(!stdr::sized_range<fs::path const>);
static_assert(!stdr::borrowed_range<fs::path const>);
static_assert(!stdr::viewable_range<fs::path const>);
