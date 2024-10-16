//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form arrays of object types.

// UNSUPPORTED: c++03

#include <array>

std::array<const int, 1> A1{0}; // okay
std::array<volatile int, 1> A2; // okay

std::array<int&, 1> A3{};
std::array<int&&, 1> A4{};
// expected-error@*:* 2 {{'std::array' cannot hold references}}

std::array<int(), 1> A5;
std::array<int(int), 1> A6;
std::array<int(int, int), 1> A7;
// expected-error@*:* 3 {{'std::array' cannot hold functions}}

std::array<void, 1> A8;
// expected-error@*:*{{'std::array' cannot hold 'void'}}

std::array<int[], 1> A9;
// expected-error@*:*{{'std::array' cannot hold C arrays of an unknown size}}

std::array<int[2], 1> A10; // okay
