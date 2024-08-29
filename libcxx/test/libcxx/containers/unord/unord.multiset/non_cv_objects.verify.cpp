//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form unordered_multisets of object types.

#include <unordered_set>

std::unordered_multiset<const int> C1;
// expected-error@*:*{{'std::unordered_multiset' cannot hold const types}}

std::unordered_multiset<volatile int> C2;
// expected-error@*:*{{'std::unordered_multiset' cannot hold volatile types}}

std::unordered_multiset<int&> C3;
std::unordered_multiset<int&&> C4;
// expected-error@*:* 2 {{'std::unordered_multiset' cannot hold references}}

std::unordered_multiset<int()> C5;
std::unordered_multiset<int(int)> C6;
std::unordered_multiset<int(int, int)> C7;
// expected-error@*:* 3 {{'std::unordered_multiset' cannot hold functions}}

std::unordered_multiset<void> C8;
// expected-error@*:*{{'std::unordered_multiset' cannot hold 'void'}}

std::unordered_multiset<int[]> C9;
std::unordered_multiset<int[2]> C10;
// expected-error@*:* 2 {{'std::unordered_multiset' cannot hold C arrays}}
