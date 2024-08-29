//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form sets of object types.

#include <set>

std::set<const int> C1;
// expected-error@*:*{{'std::set' cannot hold const types}}

std::set<volatile int> C2;
// expected-error@*:*{{'std::set' cannot hold volatile types}}

std::set<int&> C3;
std::set<int&&> C4;
// expected-error@*:* 2 {{'std::set' cannot hold references}}

std::set<int()> C5;
std::set<int(int)> C6;
std::set<int(int, int)> C7;
// expected-error@*:* 3 {{'std::set' cannot hold functions}}

std::set<void> C8;
// expected-error@*:*{{'std::set' cannot hold 'void'}}

std::set<int[]> C9;
std::set<int[2]> C10;
// expected-error@*:* 2 {{'std::set' cannot hold C arrays}}
