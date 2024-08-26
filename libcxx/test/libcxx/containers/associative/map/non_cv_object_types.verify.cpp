//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form maps of object types.

#include <map>

std::map<const int, int> K1;
std::map<volatile int, int> K2;
std::map<int&, int>
    K3; // expected-error@*:*{{'std::map::key_type' can only hold object types; references are not objects}}
std::map<int const&, int>
    K4; // expected-error@*:*{{'std::map::key_type' can only hold object types; references are not objects}}
std::map<int&&, int> K5;
std::map<int const&&, int> K6;
std::map<int(), int>
    K7; // expected-error@*:*{{'std::map::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::map<int(int), int>
    K8; // expected-error@*:*{{'std::map::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::map<int(int, int), int>
    K9; // expected-error@*:*{{'std::map::key_type' can only hold object types; functions are not objects (consider using a function pointer)}}
std::map<int (&)(), int>
    K10; // expected-error@*:*{{'std::map::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::map<int (&)(int), int>
    K11; // expected-error@*:*{{'std::map::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::map<int (&)(int, int), int>
    K12; // expected-error@*:*{{'std::map::key_type' can only hold object types; function references are not objects (consider using a function pointer)}}
std::map<int (&&)(), int> K13;
std::map<int (&&)(int), int> K14;
std::map<int (&&)(int, int), int> K15;
std::map<void, int>
    K16; // expected-error@*:*{{'std::map::key_type' can only hold object types; 'void' is not an object}}

std::map<int, const int> M1;
std::map<int, volatile int> M2;
std::map<int, int&> M3;
std::map<int, int&&> M4;
std::map<int, int()>
    M5; // expected-error@*:*{{'std::map::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::map<int, int(int)>
    M6; // expected-error@*:*{{'std::map::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::map<int, int(int, int)>
    M7; // expected-error@*:*{{'std::map::mapped_type' can only hold object or reference types; functions are neither (consider using a function pointer or reference)}}
std::map<int, int (&)()> M8;
std::map<int, int (&)(int)> M9;
std::map<int, int (&)(int, int)> M10;
std::map<int, void>
    M11; // expected-error@*:*{{'std::map::mapped_type' can only hold object types; 'void' is not an object}}
