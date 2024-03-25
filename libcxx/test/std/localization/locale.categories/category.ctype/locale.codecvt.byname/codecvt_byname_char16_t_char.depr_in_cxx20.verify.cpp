//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// codecvt_byname<char16_t, char, mbstate_t>
//  deprecated in C++20

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <locale>

#include "../with_public_dtor.hpp"

void f() {
    // Don't test for the exact type since the underlying type of
    // std::mbstate_t depends on implementation details.
    with_public_dtor<std::codecvt_byname<char16_t, char, std::mbstate_t>> cvt("", 0); // expected-warning-re {{'codecvt_byname<char16_t, char, {{.*}}>' is deprecated}}
    (void)cvt;
}
