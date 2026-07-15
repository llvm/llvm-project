//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <chrono>
//
// template<class T> struct is_clock;
// template<class T> constexpr bool is_clock_v = is_clock<T>::value;

// [time.traits.is.clock]/3:
//  The behavior of a program that adds specializations for is_clock is undefined.

// [namespace.std]/3:
//   The behavior of a C++ program is undefined if it declares an explicit or partial specialization of any standard
//   library variable template, except where explicitly permitted by the specification of that variable template.

#include <chrono>
#include <ratio>

#if !__has_warning("-Winvalid-specializations")
// expected-no-diagnostics
#else

template <>
struct std::chrono::is_clock<int> : std::false_type {}; // expected-error@*:* {{'is_clock' cannot be specialized}}

template <>
constexpr bool std::chrono::is_clock_v<float> = false; // expected-error@*:* {{'is_clock_v' cannot be specialized}}

#endif
