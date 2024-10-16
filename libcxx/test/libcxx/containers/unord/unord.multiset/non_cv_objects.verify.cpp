//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form unordered_multisets of object types.

#include <unordered_set>
#include "test_macros.h"

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
// expected-error@*:*{{'std::unordered_multiset' cannot hold C arrays of an unknown size}}

// std::hash doesn't work with C arrays, so we need to test it with something else to ensure the
// correct diagnostic is issued.
template <class T>
struct test_hash {
  using argument_type = T;
  using result_type   = std::size_t;

  result_type operator()(T const&) const;
};

std::unordered_multiset<int[2], test_hash<int[2]> > C10;
#if TEST_STD_VER < 20
// expected-error@*:*{{'std::unordered_multiset' cannot hold C arrays before C++20}}
#endif

// Spurious errors
// expected-error@__hash_table:* 1+ {{}}
// expected-error@*:* 0+ {{call to implicitly-deleted}}
// expected-error@*:* 0+ {{call to deleted}}
// expected-error@*:* 1+ {{cannot form a reference to 'void'}}
// expected-error@*:* 1+ {{declared as a pointer}}
// expected-error@*:* 1+ {{multiple overloads of}}
// expected-error@*:* 1+ {{no matching function}}
// expected-error@*:* 1+ {{no member named 'rebind'}}
// expected-error@*:* 1+ {{no type named 'const_iterator'}}
// expected-error@*:* 1+ {{no type named 'const_local_iterator'}}
// expected-error@*:* 1+ {{'std::allocator' cannot allocate}}
