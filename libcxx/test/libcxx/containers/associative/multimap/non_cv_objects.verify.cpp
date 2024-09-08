//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form multimaps of object types.

#include <map>

std::multimap<int const, int> K1;
std::multimap<int, int const> M1;
// TODO(#106635): turn this into a compile-time error

std::multimap<int volatile, int> K2;
std::multimap<int, int volatile> M2;
// TODO(#106635): turn this into a compile-time error

std::multimap<int&, int> K3;
std::multimap<int, int&> M3; // TODO(#106635): turn this into a compile-time error
// expected-error@*:* 1 {{'std::multimap' cannot hold references}}

std::multimap<int&&, int> K4;
std::multimap<int, int&&> M4;
// TODO(#106635): turn this into a compile-time error

std::multimap<int(), int> K5;
std::multimap<int(int), int> K6;
std::multimap<int(int, int), int> K7;
std::multimap<int, int()> M5;
std::multimap<int, int(int)> M6;
std::multimap<int, int(int, int)> M7;
// expected-error@*:* 6 {{'std::multimap' cannot hold functions}}

std::multimap<void, int> K8;
std::multimap<int, void> M8;
// expected-error@*:* 2 {{'std::multimap' cannot hold 'void'}}

std::multimap<int[], int> K9;
std::multimap<int, int[]> M9; // TODO(#106635): turn this into a compile-time error
// expected-error@*:*{{'std::multimap' cannot hold C arrays of an unknown size}}

std::multimap<int[2], int> K10;
std::multimap<int, int[2]> M10;
// TODO(#106635): turn this into a compile-time error
