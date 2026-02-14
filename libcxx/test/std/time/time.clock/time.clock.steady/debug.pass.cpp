//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-monotonic-clock

// <chrono>
#include <chrono>
#include <cassert>

#include "test_macros.h"

#if defined(_LIBCPP_WIN32API)
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRA_LEAN
#  include <windows.h>
#  if _WIN32_WINNT >= _WIN32_WINNT_WIN8
#    include <winapifamily.h>
#  endif
#endif // defined(_LIBCPP_WIN32API)

#if defined(_LIBCPP_WIN32API)

#  include <iostream>

// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644905(v=vs.85).aspx says:
//    If the function fails, the return value is zero. <snip>
//    On systems that run Windows XP or later, the function will always succeed
//      and will thus never return zero.

static LARGE_INTEGER __QueryPerformanceFrequency() {
  LARGE_INTEGER val;
  (void)QueryPerformanceFrequency(&val);
  return val;
}

static std::chrono::steady_clock::time_point __libcpp_steady_clock_now() {
  static const LARGE_INTEGER freq = __QueryPerformanceFrequency();

  LARGE_INTEGER counter;
  (void)QueryPerformanceCounter(&counter);
  auto seconds   = counter.QuadPart / freq.QuadPart;
  auto fractions = counter.QuadPart % freq.QuadPart;
  auto dur       = seconds * std::nano::den + fractions * std::nano::den / freq.QuadPart;
  std::cerr << "counter.QuadPart: " << counter.QuadPart << ", freq.QuadPart: " << freq.QuadPart
            << "\n seconds: " << seconds << ", fractions: " << fractions << "\n nano::den: " << std::nano::den
            << ", dur: " << dur << std::endl;
  return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(dur));
}

int main(int, char**) {
  int iteration = 100;
  for (int i = 0; i < iteration; ++i) {
    std::cerr << "Iteration: " << i << std::endl;
    auto t1 = __libcpp_steady_clock_now();
    auto t2 = __libcpp_steady_clock_now();
    assert(t2 >= t1);
    // make sure t2 didn't wrap around
  }

  return 1;
}

#else

int main(int, char**) { return 0; }

#endif
