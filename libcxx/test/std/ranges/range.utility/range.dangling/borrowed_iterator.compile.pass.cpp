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

// std::ranges::borrowed_iterator_t;

#include <ranges>

#include <concepts>
#include <span>
#include <string>
#include <string_view>
#include <vector>

static_assert(std::same_as<std::ranges::borrowed_iterator_t<std::string>, std::ranges::dangling>);
static_assert(std::same_as<std::ranges::borrowed_iterator_t<std::string&&>, std::ranges::dangling>);
static_assert(std::same_as<std::ranges::borrowed_iterator_t<std::vector<int> >, std::ranges::dangling>);

static_assert(std::same_as<std::ranges::borrowed_iterator_t<std::string&>, std::string::iterator>);
static_assert(std::same_as<std::ranges::borrowed_iterator_t<std::string_view>, std::string_view::iterator>);
static_assert(std::same_as<std::ranges::borrowed_iterator_t<std::span<int> >, std::span<int>::iterator>);

template <class T>
constexpr bool has_borrowed_iterator = requires {
  typename std::ranges::borrowed_iterator_t<T>;
};

static_assert(!has_borrowed_iterator<int>);
