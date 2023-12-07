//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// template <>
// inline constexpr bool ranges::enable_borrowed_range<filesystem::directory_iterator> = true;
// template <>
// inline constexpr bool ranges::enable_borrowed_range<filesystem::recursive_directory_iterator> = true;

#include <filesystem>
#include <ranges>

template<class Range>
void test() {
  static_assert(std::ranges::enable_borrowed_range<Range>);
  static_assert(!std::ranges::enable_borrowed_range<Range&>);
  static_assert(!std::ranges::enable_borrowed_range<const Range>);
}

void test() {
  test<std::filesystem::directory_iterator>();
  test<std::filesystem::recursive_directory_iterator>();
}
