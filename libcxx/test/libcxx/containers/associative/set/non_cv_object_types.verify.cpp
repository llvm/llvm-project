//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form sets of object types.

#include <set>
#include "test_macros.h"

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
// expected-error@*:*{{'std::set' cannot hold C arrays of an unknown size}}

std::set<int[2]> C10;
#if TEST_STD_VER < 20
// expected-error@*:*{{'std::set' cannot hold C arrays before C++20}}
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
