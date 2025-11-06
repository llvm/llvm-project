//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

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
