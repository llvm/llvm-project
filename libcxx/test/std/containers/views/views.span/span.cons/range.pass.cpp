//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

//  template<class R>
//    constexpr explicit(Extent != dynamic_extent) span(R&& r);


#include <span>
#include <cassert>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <vector>

#include "test_iterators.h"

template <class T, std::size_t Extent>
constexpr void test_from_range() {
  T val[3]{};
  std::span<T, Extent> s{val};
  assert(s.size() == std::size(val));
  assert(s.data() == std::data(val));
}

struct A {};

constexpr bool test() {
  test_from_range<int, std::dynamic_extent>();
  test_from_range<int, 3>();
  test_from_range<A, std::dynamic_extent>();
  test_from_range<A, 3>();
  return true;
}

static_assert(!std::is_constructible_v<std::span<int>, std::vector<float>&>);    // wrong type
static_assert(!std::is_constructible_v<std::span<int, 3>, std::vector<float>&>); // wrong type

static_assert(std::is_constructible_v<std::span<int>, std::vector<int>&>);                // non-borrowed lvalue
static_assert(std::is_constructible_v<std::span<int, 3>, std::vector<int>&>);             // non-borrowed lvalue
static_assert(std::is_constructible_v<std::span<const int>, std::vector<int>&>);          // non-borrowed lvalue
static_assert(std::is_constructible_v<std::span<const int, 3>, std::vector<int>&>);       // non-borrowed lvalue
static_assert(!std::is_constructible_v<std::span<int>, const std::vector<int>&>);         // non-borrowed const lvalue
static_assert(!std::is_constructible_v<std::span<int, 3>, const std::vector<int>&>);      // non-borrowed const lvalue
static_assert(std::is_constructible_v<std::span<const int>, const std::vector<int>&>);    // non-borrowed const lvalue
static_assert(std::is_constructible_v<std::span<const int, 3>, const std::vector<int>&>); // non-borrowed const lvalue
static_assert(std::is_constructible_v<std::span<const int>, std::vector<int>>);           // non-borrowed rvalue
static_assert(std::is_constructible_v<std::span<const int, 3>, std::vector<int>>);        // non-borrowed rvalue
static_assert(!std::is_constructible_v<std::span<int>, std::vector<int>&&>);              // non-borrowed rvalue
static_assert(!std::is_constructible_v<std::span<int, 3>, std::vector<int>&&>);           // non-borrowed rvalue

static_assert(std::is_constructible_v<std::span<int>, std::ranges::subrange<contiguous_iterator<int*>>>);         // contiguous borrowed rvalue
static_assert(std::is_constructible_v<std::span<int, 3>, std::ranges::subrange<contiguous_iterator<int*>>>);      // contiguous borrowed rvalue
static_assert(!std::is_constructible_v<std::span<int>, std::ranges::subrange<random_access_iterator<int*>>>);     // non-contiguous borrowed rvalue
static_assert(!std::is_constructible_v<std::span<int, 3>, std::ranges::subrange<random_access_iterator<int*>>>);  // non-contiguous borrowed rvalue

using BorrowedContiguousSizedRange = std::string_view;
static_assert(std::is_constructible_v<std::span<const char>, BorrowedContiguousSizedRange>);
static_assert(std::is_constructible_v<std::span<const char, 3>, BorrowedContiguousSizedRange>);
static_assert(!std::is_constructible_v<std::span<char>, BorrowedContiguousSizedRange>);
static_assert(!std::is_constructible_v<std::span<char, 3>, BorrowedContiguousSizedRange>);

static_assert(std::is_convertible_v<BorrowedContiguousSizedRange&, std::span<const char>>);
static_assert(!std::is_convertible_v<BorrowedContiguousSizedRange&, std::span<const char, 3>>);
static_assert(!std::is_convertible_v<BorrowedContiguousSizedRange&, std::span<char>>);
static_assert(!std::is_convertible_v<BorrowedContiguousSizedRange&, std::span<char, 3>>);
static_assert(std::is_convertible_v<const BorrowedContiguousSizedRange&, std::span<const char>>);
static_assert(!std::is_convertible_v<const BorrowedContiguousSizedRange&, std::span<const char, 3>>);

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
