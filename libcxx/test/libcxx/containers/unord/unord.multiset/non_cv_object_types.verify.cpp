//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form unordered_multisets of object types.

#include <unordered_set>

std::unordered_multiset<const int> C1; // expected-error@*:* {{'std::unordered_multiset' can only hold non-const types}}
std::unordered_multiset<volatile int>
    C2; // expected-error@*:* {{'std::unordered_multiset' can only hold non-volatile types}}
std::unordered_multiset<int&>
    C3; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; references are not objects}}
std::unordered_multiset<int&&>
    C4; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; references are not objects}}
std::unordered_multiset<int()>
    C5; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; functions are not objects (consider using a function pointer)}}
std::unordered_multiset<int(int)>
    C6; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; functions are not objects (consider using a function pointer)}}
std::unordered_multiset<int(int, int)>
    C7; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; functions are not objects (consider using a function pointer)}}
std::unordered_multiset<int (&)()>
    C8; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; function references are not objects (consider using a function pointer)}}
std::unordered_multiset<int (&)(int)>
    C9; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; function references are not objects (consider using a function pointer)}}
std::unordered_multiset<int (&)(int, int)>
    C10; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; function references are not objects (consider using a function pointer)}}
std::unordered_multiset<void>
    C11; // expected-error@*:*{{'std::unordered_multiset' can only hold object types; 'void' is not an object}}
