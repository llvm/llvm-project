//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// [[nodiscard]] stop_token get_token() const noexcept;
// [[nodiscard]] bool stop_possible() const noexcept;
// [[nodiscard]] bool stop_requested() const noexcept;
// [[nodiscard]] friend bool operator==(const stop_source& lhs, const stop_source& rhs) noexcept;

#include <stop_token>

void test() {
  std::stop_source ss;
  ss.get_token();      // expected-warning {{ignoring return value of function}}
  ss.stop_requested(); // expected-warning {{ignoring return value of function}}
  ss.stop_possible();  // expected-warning {{ignoring return value of function}}
  operator==(ss, ss);  // expected-warning {{ignoring return value of function}}
}
