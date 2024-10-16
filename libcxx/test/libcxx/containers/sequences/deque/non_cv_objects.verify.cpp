//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form deques of object types.

#include <deque>
#include "test_macros.h"

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
// expected-error@*:*{{'std::deque' cannot hold C arrays of an unknown size}}

std::deque<int[2]> C10;
#if TEST_STD_VER < 20
// expected-error@*:*{{'std::deque' cannot hold C arrays before C++20}}
#endif

// Spurious errors
// expected-error@*:* 0+ {{[allocator.requirements]}}
// expected-error@*:* 0+ {{_Tp is a function type}}
// expected-error@*:* 0+ {{arithmetic on}}
// expected-error@*:* 0+ {{assigning to}}
// expected-error@*:* 0+ {{cannot initialize a variable of type}}
// expected-error@*:* 0+ {{indirection}}
// expected-error@*:* 0+ {{object expression of non-scalar type}}
// expected-error@*:* 1+ {{'std::allocator' cannot allocate}}
// expected-error@*:* 1+ {{cannot form a reference to 'void'}}
// expected-error@*:* 1+ {{declared as a pointer}}
// expected-error@*:* 1+ {{invalid application of 'sizeof'}}
// expected-error@*:* 1+ {{multiple overloads of}}
// expected-error@*:* 1+ {{no matching function}}
// expected-error@*:* 1+ {{no member named}}
// expected-error@*:* 1+ {{no type named 'type' in 'std::enable_if}}
