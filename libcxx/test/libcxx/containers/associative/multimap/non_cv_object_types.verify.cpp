//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form multimaps of object types.

#include <map>

std::multimap<const int, int> K1;
std::multimap<volatile int, int> K2;
std::multimap<int&, int>
    K3; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; references are not objects}}
std::multimap<int const&, int>
    K4; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; references are not objects}}
std::multimap<int&&, int> K5;
std::multimap<int const&&, int> K6;
std::multimap<int(), int>
    K7; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::multimap<int(int), int>
    K8; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::multimap<int(int, int), int>
    K9; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::multimap<int (&)(), int>
    K10; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::multimap<int (&)(int), int>
    K11; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::multimap<int (&)(int, int), int>
    K12; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::multimap<int (&&)(), int> K13;
std::multimap<int (&&)(int), int> K14;
std::multimap<int (&&)(int, int), int> K15;
std::multimap<void, int>
    K16; // expected-error@*:*{{'std::multimap::key_type' can only hold object types; 'void' is not an object}}

std::multimap<int, const int> M1;
std::multimap<int, volatile int> M2;
std::multimap<int, int&> M3;
std::multimap<int, int&&> M4;
std::multimap<int, int()>
    M5; // expected-error@*:*{{'std::multimap::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::multimap<int, int(int)>
    M6; // expected-error@*:*{{'std::multimap::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::multimap<int, int(int, int)>
    M7; // expected-error@*:*{{'std::multimap::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::multimap<int, int (&)()> M8;
std::multimap<int, int (&)(int)> M9;
std::multimap<int, int (&)(int, int)> M10;
std::multimap<int, void>
    M11; // expected-error@*:*{{'std::multimap::mapped_type' can only hold object types; 'void' is not an object}}
