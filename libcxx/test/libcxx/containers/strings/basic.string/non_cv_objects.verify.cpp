//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form basic_strings of object types.

#include <string>

std::basic_string<const int> C1;    // expected-error@*:* {{'std::basic_string' can only hold non-const types}}
std::basic_string<volatile int> C2; // expected-error@*:* {{'std::basic_string' can only hold non-volatile types}}
std::basic_string<int&>
    C3; // expected-error@*:*{{'std::basic_string' can only hold object types; references are not objects}}
std::basic_string<int&&>
    C4; // expected-error@*:*{{'std::basic_string' can only hold object types; references are not objects}}
std::basic_string<int()>
    C5; // expected-error@*:*{{'std::basic_string' can only hold object types; functions are not objects (consider using a function pointer)}}
std::basic_string<int(int)>
    C6; // expected-error@*:*{{'std::basic_string' can only hold object types; functions are not objects (consider using a function pointer)}}
std::basic_string<int(int, int)>
    C7; // expected-error@*:*{{'std::basic_string' can only hold object types; functions are not objects (consider using a function pointer)}}
std::basic_string<int (&)()>
    C8; // expected-error@*:*{{'std::basic_string' can only hold object types; function references are not objects (consider using a function pointer)}}
std::basic_string<int (&)(int)>
    C9; // expected-error@*:*{{'std::basic_string' can only hold object types; function references are not objects (consider using a function pointer)}}
std::basic_string<int (&)(int, int)>
    C10; // expected-error@*:*{{'std::basic_string' can only hold object types; function references are not objects (consider using a function pointer)}}
std::basic_string<void>
    C11; // expected-error@*:*{{'std::basic_string' can only hold object types; 'void' is not an object}}
