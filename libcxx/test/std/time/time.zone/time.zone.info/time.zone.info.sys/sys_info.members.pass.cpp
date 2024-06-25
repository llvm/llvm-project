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

//  struct sys_info {
//    sys_seconds   begin;
//    sys_seconds   end;
//    seconds       offset;
//    minutes       save;
//    string        abbrev;
//  };

// Validates whether:
// - The members are present as non-const members.
// - The struct is an aggregate.

#include <chrono>
#include <string>
#include <type_traits>

int main(int, const char**) {
  static_assert(std::is_aggregate_v<std::chrono::sys_info>);

  std::chrono::sys_info sys_info{
      .begin  = std::chrono::sys_seconds::min(),
      .end    = std::chrono::sys_seconds::max(),
      .offset = std::chrono::seconds(0),
      .save   = std::chrono::minutes(0),
      .abbrev = "UTC"};

  [[maybe_unused]] std::chrono::sys_seconds& begin = sys_info.begin;
  [[maybe_unused]] std::chrono::sys_seconds& end   = sys_info.end;
  [[maybe_unused]] std::chrono::seconds& offset    = sys_info.offset;
  [[maybe_unused]] std::chrono::minutes& save      = sys_info.save;
  [[maybe_unused]] std::string& abbrev             = sys_info.abbrev;

  return 0;
}
