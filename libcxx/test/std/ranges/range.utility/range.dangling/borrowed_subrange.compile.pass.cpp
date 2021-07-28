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

// std::ranges::borrowed_subrange_t;

#include <ranges>

#include <concepts>
#include <span>
#include <string>
#include <string_view>
#include <vector>

static_assert(std::same_as<std::ranges::borrowed_subrange_t<std::string>, std::ranges::dangling>);
static_assert(std::same_as<std::ranges::borrowed_subrange_t<std::string&&>, std::ranges::dangling>);
static_assert(std::same_as<std::ranges::borrowed_subrange_t<std::vector<int> >, std::ranges::dangling>);

static_assert(
    std::same_as<std::ranges::borrowed_subrange_t<std::string&>, std::ranges::subrange<std::string::iterator> >);

static_assert(
    std::same_as<std::ranges::borrowed_subrange_t<std::span<int> >, std::ranges::subrange<std::span<int>::iterator> >);

static_assert(std::same_as<std::ranges::borrowed_subrange_t<std::string_view>,
                           std::ranges::subrange<std::string_view::iterator> >);

template <class T>
constexpr bool has_type = requires {
  typename std::ranges::borrowed_subrange_t<T>;
};

static_assert(!has_type<int>);

struct S {};
static_assert(!has_type<S>);
