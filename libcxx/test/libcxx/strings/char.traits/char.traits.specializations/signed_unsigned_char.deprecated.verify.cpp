//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<unsigned char>
// template<> struct char_traits<signed char>

// Make sure we issue deprecation warnings.

#include <string>

void f() {
    std::char_traits<unsigned char> uc; // expected-warning{{'char_traits<unsigned char>' is deprecated}}
    std::char_traits<signed char> sc; // expected-warning{{'char_traits<signed char>' is deprecated}}

    (void)uc;
    (void)sc;
}
