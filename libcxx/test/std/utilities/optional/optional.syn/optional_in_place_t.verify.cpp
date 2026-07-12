//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// A program that necessitates the instantiation of template optional for
// (possibly cv-qualified) in_place_t is ill-formed.

#include <optional>

#include "test_macros.h"

void f() {
    std::optional<std::in_place_t> opt; // expected-note {{requested here}}
    // expected-error@*:* 1 {{instantiation of optional with in_place_t is ill-formed}}
#if TEST_STD_VER >= 26
    std::optional<std::in_place_t&> opt1; // expected-note {{requested here}}
    // expected-error@*:* 1 {{instantiation of optional with in_place_t is ill-formed}}
#endif
}
