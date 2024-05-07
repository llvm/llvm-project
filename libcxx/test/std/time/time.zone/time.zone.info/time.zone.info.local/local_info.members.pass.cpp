//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: libcpp-has-no-experimental-tzdb

// <chrono>

// struct local_info {
//   static constexpr int unique      = 0;
//   static constexpr int nonexistent = 1;
//   static constexpr int ambiguous   = 2;
//
//   int result;
//   sys_info first;
//   sys_info second;
// };

// Validates whether:
// - The static members are present as static constexpr members.
// - The members are present as non-const members.
// - The struct is an aggregate.

#include <chrono>
#include <string>
#include <type_traits>

int main(int, const char**) {
  {
    constexpr const int& result = std::chrono::local_info::unique;
    static_assert(result == 0);
  }
  {
    constexpr const int& result = std::chrono::local_info::nonexistent;
    static_assert(result == 1);
  }
  {
    constexpr const int& result = std::chrono::local_info::ambiguous;
    static_assert(result == 2);
  }

  static_assert(std::is_aggregate_v<std::chrono::local_info>);

  std::chrono::local_info local_info{.result = 0, .first = std::chrono::sys_info{}, .second = std::chrono::sys_info{}};

  [[maybe_unused]] int& result                   = local_info.result;
  [[maybe_unused]] std::chrono::sys_info& first  = local_info.first;
  [[maybe_unused]] std::chrono::sys_info& second = local_info.second;

  return 0;
}
