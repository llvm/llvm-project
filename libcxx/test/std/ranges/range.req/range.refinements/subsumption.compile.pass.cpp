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

// template<class T>
// concept input_iterator;

// std::ranges::forward_range

#include <ranges>

#include <iterator>

struct range {
  int* begin();
  int* end();
};

template<std::ranges::range R>
requires std::input_iterator<std::ranges::iterator_t<R>>
[[nodiscard]] constexpr bool check_input_range_subsumption() {
  return false;
}

template<std::ranges::input_range>
requires true
[[nodiscard]] constexpr bool check_input_range_subsumption() {
  return true;
}

static_assert(check_input_range_subsumption<range>());

template<std::ranges::input_range R>
requires std::forward_iterator<std::ranges::iterator_t<R>>
[[nodiscard]] constexpr bool check_forward_range_subsumption() {
  return false;
}

template<std::ranges::forward_range>
requires true
[[nodiscard]] constexpr bool check_forward_range_subsumption() {
  return true;
}

static_assert(check_forward_range_subsumption<range>());

template<std::ranges::forward_range R>
requires std::bidirectional_iterator<std::ranges::iterator_t<R>>
[[nodiscard]] constexpr bool check_bidirectional_range_subsumption() {
  return false;
}

template<std::ranges::bidirectional_range>
requires true
[[nodiscard]] constexpr bool check_bidirectional_range_subsumption() {
  return true;
}

static_assert(check_bidirectional_range_subsumption<range>());

template<std::ranges::bidirectional_range R>
requires std::random_access_iterator<std::ranges::iterator_t<R>>
constexpr bool check_random_access_range_subsumption() {
  return false;
}

template<std::ranges::random_access_range>
requires true
constexpr bool check_random_access_range_subsumption() {
  return true;
}

static_assert(check_random_access_range_subsumption<range>());

template<std::ranges::random_access_range R>
requires std::random_access_iterator<std::ranges::iterator_t<R>>
constexpr bool check_contiguous_range_subsumption() {
  return false;
}

template<std::ranges::contiguous_range>
requires true
constexpr bool check_contiguous_range_subsumption() {
  return true;
}

static_assert(check_contiguous_range_subsumption<range>());
