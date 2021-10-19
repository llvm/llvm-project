//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// template <class Duration> class hh_mm_ss;
//   If Duration is not an instance of duration, the program is ill-formed.

#include <chrono>
#include <string>
#include <cassert>
#include "test_macros.h"

struct A {};

int main(int, char**)
{
    std::chrono::hh_mm_ss<void> h0;        // expected-error-re@chrono:* {{static_assert failed {{.*}} "template parameter of hh_mm_ss must be a std::chrono::duration"}}
    std::chrono::hh_mm_ss<int> h1;         // expected-error-re@chrono:* {{static_assert failed {{.*}} "template parameter of hh_mm_ss must be a std::chrono::duration"}}
    std::chrono::hh_mm_ss<std::string> h2; // expected-error-re@chrono:* {{static_assert failed {{.*}} "template parameter of hh_mm_ss must be a std::chrono::duration"}}
    std::chrono::hh_mm_ss<A> h3;           // expected-error-re@chrono:* {{static_assert failed {{.*}} "template parameter of hh_mm_ss must be a std::chrono::duration"}}

    return 0;
}
