//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form deques of object types.

#include <deque>

std::deque<const int> C1;
// expected-error@*:*{{'std::deque' cannot hold const types}}

std::deque<volatile int> C2;
// expected-error@*:*{{'std::deque' cannot hold volatile types}}

std::deque<int&> C3;
std::deque<int&&> C4;
// expected-error@*:* 2 {{'std::deque' cannot hold references}}

std::deque<int()> C5;
std::deque<int(int)> C6;
std::deque<int(int, int)> C7;
// expected-error@*:* 3 {{'std::deque' cannot hold functions}}

std::deque<void> C8;
// expected-error@*:*{{'std::deque' cannot hold 'void'}}

std::deque<int[]> C9;
std::deque<int[2]> C10;
// expected-error@*:* 2 {{'std::deque' cannot hold C arrays}}
