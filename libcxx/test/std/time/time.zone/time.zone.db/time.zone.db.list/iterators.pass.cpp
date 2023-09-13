//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// TODO TZDB Enable tests
// UNSUPPORTED: c++20, c++23, c++26

// <chrono>
//
// class tzdb_list;
//
// const_iterator begin() const noexcept;
// const_iterator end()   const noexcept;
//
// const_iterator cbegin() const noexcept;
// const_iterator cend()   const noexcept;

#include <chrono>
#include <iterator>
#include <cassert>

int main(int, char**) {
  const std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
  using it                           = std::chrono::tzdb_list::const_iterator;

  static_assert(noexcept(list.begin()));
  static_assert(noexcept(list.end()));
  static_assert(noexcept(list.cbegin()));
  static_assert(noexcept(list.cend()));

  std::same_as<it> auto begin = list.begin();
  std::same_as<it> auto end   = list.end();
  assert(std::distance(begin, end) == 1);

  std::same_as<it> auto cbegin = list.cbegin();
  assert(begin == cbegin);

  std::same_as<it> auto cend = list.cend();
  assert(end == cend);

  return 0;
}
