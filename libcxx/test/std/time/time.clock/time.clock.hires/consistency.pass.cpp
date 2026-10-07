//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: std-at-least-c++11
//
// Due to C++17 inline variables ASAN flags this test as containing an ODR
// violation because Clock::is_steady is defined in both the dylib and this TU.
// UNSUPPORTED: asan

// <chrono>

// high_resolution_clock

// check clock invariants

#include <cassert>
#include <chrono>
#include <type_traits>

#include "test_macros.h"

void odr_use(const bool&) {}

int main(int, char**)
{
    typedef std::chrono::high_resolution_clock C;
    static_assert((std::is_same<C::rep, C::duration::rep>::value), "");
    static_assert((std::is_same<C::period, C::duration::period>::value), "");
    static_assert((std::is_same<C::duration, C::time_point::duration>::value), "");

    static_assert(std::is_same<decltype(C::is_steady), const bool>::value, "is_steady must be bool");
    static_assert(!std::is_member_pointer<decltype(&C::is_steady)>::value, "is_steady must be static");
    TEST_CONSTEXPR_CXX14 const bool is_steady = C::is_steady; // "is_steady must be constexpr"
    (void)is_steady;

// high_resolution_clock typedefs to steady_clock when _LIBCPP_HAS_MONOTONIC_CLOCK,
// else system_clock (e.g. generic-llvm-libc, generic-no-threads).
#if _LIBCPP_HAS_MONOTONIC_CLOCK
  LIBCPP_ASSERT(C::is_steady);
#else
  LIBCPP_ASSERT(!C::is_steady);
#endif

  odr_use(C::is_steady);

  return 0;
}
