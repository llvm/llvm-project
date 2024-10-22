//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form vectors of object types.

#include <vector>
#include "test_macros.h"

std::vector<const int> C1;
// expected-error@*:*{{'std::vector' cannot hold const types}}

std::vector<volatile int> C2;
// expected-error@*:*{{'std::vector' cannot hold volatile types}}

std::vector<int&> C3;
std::vector<int&&> C4;
// expected-error@*:* 2 {{'std::vector' cannot hold references}}

std::vector<int()> C5;
std::vector<int(int)> C6;
std::vector<int(int, int)> C7;
// expected-error@*:* 3 {{'std::vector' cannot hold functions}}

std::vector<void> C8;
// expected-error@*:*{{'std::vector' cannot hold 'void'}}

std::vector<int[]> C9;
// expected-error@*:*{{'std::vector' cannot hold C arrays of an unknown size}}

std::vector<int[2]> C10;
#if TEST_STD_VER < 20
// expected-error@*:*{{'std::vector' cannot hold C arrays before C++20}}
#endif

// Bogus errors
// expected-error@*:* 0+ {{_Tp is a function type}}
// expected-error@*:* 0+ {{[allocator.requirements]}}
// expected-error@*:* 0+ {{cannot initialize a variable of type}}
// expected-error@*:* 0+ {{object expression of non-scalar type}}
// expected-error@*:* 0+ {{cannot initialize a parameter of type 'const void *'}}
// expected-error@*:* 1+ {{'std::allocator' cannot allocate}}
// expected-error@*:* 1+ {{arithmetic on}}
// expected-error@*:* 1+ {{cannot form a reference to 'void'}}
// expected-error@*:* 1+ {{declared as a pointer}}
// expected-error@*:* 1+ {{invalid application of 'sizeof'}}
// expected-error@*:* 1+ {{multiple overloads of}}
// expected-error@*:* 1+ {{no matching function}}
// expected-error@*:* 1+ {{no member named 'rebind'}}
// expected-error@*:* 1+ {{no type named}}
// expected-warning@*:* 0+ {{is a Microsoft extension}}
