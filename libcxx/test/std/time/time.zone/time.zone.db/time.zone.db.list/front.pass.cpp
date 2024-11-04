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

// TODO TZDB (#81654) Enable tests
// UNSUPPORTED: c++20, c++23, c++26

// <chrono>
//
// class tzdb_list;
//
// const tzdb& front() const noexcept;

#include <chrono>

int main(int, char**) {
  const std::chrono::tzdb_list& list          = std::chrono::get_tzdb_list();
  [[maybe_unused]] const std::chrono::tzdb& _ = list.front();
  static_assert(noexcept(list.front()));

  return 0;
}
