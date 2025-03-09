//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>
//
// class utc_clock;

// static time_point now();

#include <chrono>
#include <concepts>
#include <cassert>

int main(int, const char**) {
  using clock                                      = std::chrono::utc_clock;
  std::same_as<clock::time_point> decltype(auto) t = clock::now();

  assert(t >= clock::time_point::min());
  assert(t <= clock::time_point::max());

  auto t2 = clock::now();
  assert(t2 - t >= std::chrono::seconds(0));
  // This may fail if the tests takes a long time to complete.
  assert(t2 - t < std::chrono::seconds(42));

  return 0;
}
