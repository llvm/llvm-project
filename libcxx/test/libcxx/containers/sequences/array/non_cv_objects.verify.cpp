//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form arrays of object types.

#include <array>

std::array<const int, 1> A1{};  // okay
std::array<volatile int, 1> A2; // okay

int I;

std::array<int&, 1> A3{I}; // expected-error@*:*{{'std::array' can only hold object types; references are not objects}}
std::array<int&&, 1> A4{
    static_cast<int&&>(I)}; // expected-error@*:*{{'std::array' can only hold object types; references are not objects}}
std::array<int(), 1>
    A5; // expected-error@*:*{{'std::array' can only hold object types; functions are not objects (consider using a function pointer)}}
std::array<int (&)(), 1>
    A6; // expected-error@*:*{{'std::array' can only hold object types; function references are not objects (consider using a function pointer)}}
std::array<void, 1> A7; // expected-error@*:*{{'std::array' can only hold object types; 'void' is not an object}}
