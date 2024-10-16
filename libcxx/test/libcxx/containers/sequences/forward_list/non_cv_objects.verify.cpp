//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form forward_lists of object types.

#include <forward_list>
#include "test_macros.h"

std::forward_list<const int> C1;
// expected-error@*:*{{'std::forward_list' cannot hold const types}}

std::forward_list<volatile int> C2;
// expected-error@*:*{{'std::forward_list' cannot hold volatile types}}

std::forward_list<int&> C3;
std::forward_list<int&&> C4;
// expected-error@*:* 2 {{'std::forward_list' cannot hold references}}

std::forward_list<int()> C5;
std::forward_list<int(int)> C6;
std::forward_list<int(int, int)> C7;
// expected-error@*:* 3 {{'std::forward_list' cannot hold functions}}

std::forward_list<void> C8;
// expected-error@*:*{{'std::forward_list' cannot hold 'void'}}

std::forward_list<int[]> C9;
// expected-error@*:*{{'std::forward_list' cannot hold C arrays of an unknown size}}

std::forward_list<int[2]> C10;
#if TEST_STD_VER < 20
// expected-error@*:*{{'std::forward_list' cannot hold C arrays before C++20}}
#endif

// Spurious errors
// expected-error@*:* 0+ {{[allocator.requirements]}}
// expected-error@*:* 0+ {{object expression of non-scalar type}}
// expected-error@*:* 0+ {{data member instantiated}}
// expected-error@*:* 0+ {{invalid application of 'sizeof'}}
// expected-error@*:* 1+ {{cannot form a reference to 'void'}}
// expected-error@*:* 1+ {{declared as a pointer}}
// expected-error@*:* 1+ {{multiple overloads of}}
// expected-error@*:* 1+ {{no matching function}}
// expected-error@*:* 1+ {{no member named}}
// expected-error@*:* 1+ {{no type named 'type' in 'std::enable_if}}
// expected-error@*:* 1+ {{__node_}}
// expected-error@*:* 1+ {{pointer}}
// expected-error@*:* 1+ {{iterator}}
// expected-error@*:* 1+ {{'std::allocator' cannot allocate}}
