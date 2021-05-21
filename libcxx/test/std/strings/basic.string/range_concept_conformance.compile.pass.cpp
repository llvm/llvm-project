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

// string

#include <string>

#include <concepts>
#include <ranges>

namespace stdr = std::ranges;

static_assert(std::same_as<stdr::iterator_t<std::string>, std::string::iterator>);
static_assert(stdr::common_range<std::string>);
static_assert(stdr::random_access_range<std::string>);
static_assert(!stdr::view<std::string>);
static_assert(stdr::sized_range<std::string>);
static_assert(!stdr::borrowed_range<std::string>);

static_assert(std::same_as<stdr::iterator_t<std::string const>, std::string::const_iterator>);
static_assert(stdr::common_range<std::string const>);
static_assert(stdr::random_access_range<std::string const>);
static_assert(!stdr::view<std::string const>);
static_assert(stdr::sized_range<std::string const>);
static_assert(!stdr::borrowed_range<std::string const>);
