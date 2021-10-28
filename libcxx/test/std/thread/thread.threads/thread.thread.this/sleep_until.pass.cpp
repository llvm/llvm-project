//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// TODO(ldionne): This test fails on Ubuntu Focal on our CI nodes (and only there), in 32 bit mode.
// UNSUPPORTED: linux && 32bits-on-64bits

// <thread>

// template <class Clock, class Duration>
//   void sleep_until(const chrono::time_point<Clock, Duration>& abs_time);

#include <thread>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
  typedef std::chrono::system_clock Clock;
  typedef Clock::time_point time_point;
  std::chrono::milliseconds ms(500);
  time_point t0 = Clock::now();
  std::this_thread::sleep_until(t0 + ms);
  time_point t1 = Clock::now();
  // NOTE: Operating systems are (by default) best effort and therefore we may
  // have slept longer, perhaps much longer than we requested.
  assert(t1 - t0 >= ms);

  return 0;
}
