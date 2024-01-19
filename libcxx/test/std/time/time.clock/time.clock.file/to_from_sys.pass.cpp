//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// UNSUPPORTED: availability-filesystem-missing

// "unable to find library from dependent library specifier: rt"
// XFAIL: LIBCXX-PICOLIBC-FIXME

// <chrono>
//
// file_clock
//
// template<class Duration>
// static sys_time<see-below> to_sys(const file_time<Duration>&);
//
// template<class Duration>
// static file_time<see-below> from_sys(const sys_time<Duration>&);

#include <chrono>
#include <cassert>

int main(int, char**) {
  // Test round-trip through the system clock, starting from file_clock::now()
  {
    std::chrono::file_clock::time_point const ft = std::chrono::file_clock::now();
    auto st = std::chrono::file_clock::to_sys(ft);
    assert(ft == std::chrono::file_clock::from_sys(st));
  }

  // Test round-trip through the system clock, starting from system_clock::now()
  {
    std::chrono::system_clock::time_point const st = std::chrono::system_clock::now();
    auto ft = std::chrono::file_clock::from_sys(st);
    assert(st == std::chrono::file_clock::to_sys(ft));
  }

  // Make sure the value we get is in the ballpark of something reasonable
  {
    std::chrono::file_clock::time_point const file_now = std::chrono::file_clock::now();
    std::chrono::system_clock::time_point const sys_now = std::chrono::system_clock::now();
    {
      auto diff = sys_now - std::chrono::file_clock::to_sys(file_now);
      assert(std::chrono::milliseconds(-500) < diff && diff < std::chrono::milliseconds(500));
    }
    {
      auto diff = std::chrono::file_clock::from_sys(sys_now) - file_now;
      assert(std::chrono::milliseconds(-500) < diff && diff < std::chrono::milliseconds(500));
    }
  }

  // Make sure to_sys and from_sys are consistent with each other
  {
    std::chrono::file_clock::time_point const ft = std::chrono::file_clock::now();
    std::chrono::system_clock::time_point const st = std::chrono::system_clock::now();
    auto sys_diff = std::chrono::file_clock::to_sys(ft) - st;
    auto file_diff = ft - std::chrono::file_clock::from_sys(st);
    assert(sys_diff == file_diff);
  }

  return 0;
}
