//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// XFAIL: libcpp-has-no-experimental-tzdb

// <chrono>

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const local_info& r);

// [time.zone.info.local]
//   3 Effects: Streams out the local_info object r in an unspecified format.
//   4 Returns: os.
//
// There is a private libc++ test that validates the exact output.

#include <cassert>
#include <chrono>
#include <memory>
#include <sstream>

#include "test_macros.h"

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;
  std::chrono::sys_info s{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 0h, 0min, ""};
  std::chrono::local_info l{0, s, s};
  std::basic_ostringstream<CharT> os;
  std::basic_ostream<CharT>& result = std::chrono::operator<<(os, l);
  assert(std::addressof(result) == std::addressof(os));
}

int main(int, const char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
