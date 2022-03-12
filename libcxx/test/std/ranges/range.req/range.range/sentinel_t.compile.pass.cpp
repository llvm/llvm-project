//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<range _Rp>
// using sentinel_t = decltype(ranges::end(declval<_Rp&>()));

#include <ranges>

#include <concepts>

#include "test_iterators.h"
#include "test_range.h"



static_assert(std::same_as<std::ranges::sentinel_t<test_range<cpp20_input_iterator> >, sentinel>);
static_assert(std::same_as<std::ranges::sentinel_t<test_range<cpp20_input_iterator> const>, sentinel>);
static_assert(std::same_as<std::ranges::sentinel_t<test_non_const_range<cpp20_input_iterator> >, sentinel>);
static_assert(std::same_as<std::ranges::sentinel_t<test_common_range<forward_iterator> >, forward_iterator<int*> >);
static_assert(std::same_as<std::ranges::sentinel_t<test_common_range<forward_iterator> const>, forward_iterator<int const*> >);
static_assert(std::same_as<std::ranges::sentinel_t<test_non_const_common_range<forward_iterator> >, forward_iterator<int*> >);
