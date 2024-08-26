//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form multimaps of object types.

#include <unordered_map>

std::unordered_multimap<int&, int>
    K3; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; references are not objects}}
std::unordered_multimap<int&&, int>
    K4; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; references are not objects}}
std::unordered_multimap<int(), int>
    K5; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::unordered_multimap<int(int), int>
    K6; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::unordered_multimap<int(int, int), int>
    K7; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::unordered_multimap<int (&)(), int>
    K8; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::unordered_multimap<int (&)(int), int>
    K9; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::unordered_multimap<int (&)(int, int), int>
    K10; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::unordered_multimap<void, int>
    K11; // expected-error@*:*{{'std::unordered_multimap::key_type' can only hold object types; 'void' is not an object}}

std::unordered_multimap<int, const int> M1;
std::unordered_multimap<int, volatile int> M2;
std::unordered_multimap<int, int&> M3;
std::unordered_multimap<int, int&&> M4;
std::unordered_multimap<int, int()>
    M5; // expected-error@*:*{{'std::unordered_multimap::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::unordered_multimap<int, int(int)>
    M6; // expected-error@*:*{{'std::unordered_multimap::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::unordered_multimap<int, int(int, int)>
    M7; // expected-error@*:*{{'std::unordered_multimap::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::unordered_multimap<int, int (&)()> M8;
std::unordered_multimap<int, int (&)(int)> M9;
std::unordered_multimap<int, int (&)(int, int)> M10;
std::unordered_multimap<int, void>
    M11; // expected-error@*:*{{'std::unordered_multimap::mapped_type' can only hold object types; 'void' is not an object}}
