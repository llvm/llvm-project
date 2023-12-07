//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// string

#include <string>

#include <concepts>
#include <ranges>

static_assert(std::same_as<std::ranges::iterator_t<std::string>, std::string::iterator>);
static_assert(std::ranges::common_range<std::string>);
static_assert(std::ranges::random_access_range<std::string>);
static_assert(std::ranges::contiguous_range<std::string>);
static_assert(!std::ranges::view<std::string>);
static_assert(std::ranges::sized_range<std::string>);
static_assert(!std::ranges::borrowed_range<std::string>);
static_assert(std::ranges::viewable_range<std::string>);

static_assert(std::same_as<std::ranges::iterator_t<std::string const>, std::string::const_iterator>);
static_assert(std::ranges::common_range<std::string const>);
static_assert(std::ranges::random_access_range<std::string const>);
static_assert(std::ranges::contiguous_range<std::string const>);
static_assert(!std::ranges::view<std::string const>);
static_assert(std::ranges::sized_range<std::string const>);
static_assert(!std::ranges::borrowed_range<std::string const>);
static_assert(!std::ranges::viewable_range<std::string const>);
