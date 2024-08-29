//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form lists of object types.

#include <list>

std::list<const int> C1;
// expected-error@*:*{{'std::list' cannot hold const types}}

std::list<volatile int> C2;
// expected-error@*:*{{'std::list' cannot hold volatile types}}

std::list<int&> C3;
std::list<int&&> C4;
// expected-error@*:* 2 {{'std::list' cannot hold references}}

std::list<int()> C5;
std::list<int(int)> C6;
std::list<int(int, int)> C7;
// expected-error@*:* 3 {{'std::list' cannot hold functions}}

std::list<void> C8;
// expected-error@*:*{{'std::list' cannot hold 'void'}}

std::list<int[]> C9;
std::list<int[2]> C10;
// expected-error@*:* 2 {{'std::list' cannot hold C arrays}}

// Spurious errors below

// expected-error@*:* 7 {{'std::allocator'}}
// expected-error@*:* 3 {{multiple overloads of}}
// expected-error@*:* 14 {{cannot form a reference to 'void'}}
// expected-error@*:* 144 {{no type named}}
