//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>
//
// class gps_clock;

// static time_point now();

#include <chrono>
#include <concepts>
#include <cassert>

int main(int, const char**) {
  using clock                                      = std::chrono::gps_clock;
  std::same_as<clock::time_point> decltype(auto) t = clock::now();

  assert(t >= clock::time_point::min());
  assert(t <= clock::time_point::max());

  return 0;
}
