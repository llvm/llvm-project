//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form maps of object types.

#include <map>

std::map<int const, int> K1; // not an error
std::map<int, int const> M1; // not an error

std::map<int volatile, int> K2; // not an error
std::map<int, int volatile> M2; // not an error

std::map<int&, int> K3;
// expected-error@*:* {{'std::map' cannot hold references}}
std::map<int, int&> M3; // not an error

std::map<int&&, int> K4; // not an error
std::map<int, int&&> M4; // not an error

std::map<int(), int> K5;
std::map<int(int), int> K6;
std::map<int(int, int), int> K7;
std::map<int, int()> M5;
std::map<int, int(int)> M6;
std::map<int, int(int, int)> M7;
// expected-error@*:* 6 {{'std::map' cannot hold functions}}

std::map<void, int> K8;
std::map<int, void> M8;
// expected-error@*:* 2 {{'std::map' cannot hold 'void'}}

std::map<int[], int> K9;
// expected-error@*:*{{'std::map' cannot hold C arrays of an unknown size}}
std::map<int, int[]> M9; // not an error

std::map<int[2], int> K10; // not an error
std::map<int, int[2]> M10; // not an error
