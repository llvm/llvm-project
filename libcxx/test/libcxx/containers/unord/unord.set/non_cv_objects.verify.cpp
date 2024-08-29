//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form unordered_sets of object types.

#include <unordered_set>

std::unordered_set<const int> C1;
// expected-error@*:*{{'std::unordered_set' cannot hold const types}}

std::unordered_set<volatile int> C2;
// expected-error@*:*{{'std::unordered_set' cannot hold volatile types}}

std::unordered_set<int&> C3;
std::unordered_set<int&&> C4;
// expected-error@*:* 2 {{'std::unordered_set' cannot hold references}}

std::unordered_set<int()> C5;
std::unordered_set<int(int)> C6;
std::unordered_set<int(int, int)> C7;
// expected-error@*:* 3 {{'std::unordered_set' cannot hold functions}}

std::unordered_set<void> C8;
// expected-error@*:*{{'std::unordered_set' cannot hold 'void'}}

std::unordered_set<int[]> C9;
std::unordered_set<int[2]> C10;
// expected-error@*:* 2 {{'std::unordered_set' cannot hold C arrays}}
