//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// <utility>

// Expect failures for tuple_element and get with empty integer_sequence

#include <utility>

void f() {
    using test1 = typename std::tuple_element<0, std::integer_sequence<int>>::type; // expected-error-re@*:* {{static assertion failed{{.*}}Index out of bounds in std::tuple_element<> (std::integer_sequence)}}
    using test2 = typename std::tuple_element<0, const std::integer_sequence<int>>::type; // expected-error-re@*:* {{static assertion failed{{.*}}Index out of bounds in std::tuple_element<> (const std::integer_sequence)}}

    auto empty = std::integer_sequence<int>();
    // expected-error-re@*:* {{static assertion failed{{.*}}Index out of bounds in std::get<> (std::integer_sequence)}}
    // expected-error-re@*:* {{invalid index 0 for pack '{{.*}}' of size 0}}
    (void)std::get<0>(empty);
}
