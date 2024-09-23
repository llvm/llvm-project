//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form multisets of object types.

#include <set>
#include "test_macros.h"

std::multiset<const int> C1;
// expected-error@*:*{{'std::multiset' cannot hold const types}}

std::multiset<volatile int> C2;
// expected-error@*:*{{'std::multiset' cannot hold volatile types}}

std::multiset<int&> C3;
std::multiset<int&&> C4;
// expected-error@*:* 2 {{'std::multiset' cannot hold references}}

std::multiset<int()> C5;
std::multiset<int(int)> C6;
std::multiset<int(int, int)> C7;
// expected-error@*:* 3 {{'std::multiset' cannot hold functions}}

std::multiset<void> C8;
// expected-error@*:*{{'std::multiset' cannot hold 'void'}}

std::multiset<int[]> C9;
// expected-error@*:*{{'std::multiset' cannot hold C arrays of an unknown size}}

std::multiset<int[2]> C10;
#if TEST_STD_VER < 20
// expected-error@*:*{{'std::multiset' cannot hold C arrays before C++20}}
#endif

// Spurious errors
// expected-error@__tree:* 1+ {{}}
// expected-error@*:* 1+ {{cannot form a reference to 'void'}}
// expected-error@*:* 1+ {{declared as a pointer}}
// expected-error@*:* 1+ {{no matching function}}
// expected-error@*:* 1+ {{no member named 'rebind'}}
// expected-error@*:* 1+ {{std::__tree}}
// expected-error@*:* 1+ {{multiple overloads of}}
// expected-error@*:* 1+ {{'std::allocator' cannot allocate}}
