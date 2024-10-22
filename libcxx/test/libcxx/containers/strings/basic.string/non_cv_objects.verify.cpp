//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form basic_strings of object types.

#include <string>

std::basic_string<const int> C1;
// expected-error@*:*{{'std::basic_string' cannot hold const types}}

std::basic_string<volatile int> C2;
// expected-error@*:*{{'std::basic_string' cannot hold volatile types}}

std::basic_string<int&> C3;
// expected-error@*:*{{'std::basic_string' cannot hold references}}

std::basic_string<int&&> C4;
// expected-error@*:*{{'std::basic_string' cannot hold references}}

std::basic_string<int()> C5;
// expected-error@*:*{{'std::basic_string' cannot hold functions}}

std::basic_string<int(int)> C6;
// expected-error@*:*{{'std::basic_string' cannot hold functions}}

std::basic_string<int(int, int)> C7;
// expected-error@*:*{{'std::basic_string' cannot hold functions}}

std::basic_string<void> C8;
// expected-error@*:*{{'std::basic_string' cannot hold 'void'}}

std::basic_string<int[]> C9;
// expected-error@*:*{{'std::basic_string' cannot hold C arrays}}

std::basic_string<int[2]> C10;
// expected-error@*:*{{'std::basic_string' cannot hold C arrays}}

// Bogus errors
// expected-error@*:* 0+ {{[allocator.requirements] states that rebinding}}
// expected-error@*:* 0+ {{call to implicitly-deleted}}
// expected-error@*:* 0+ {{cannot initialize a variable of type}}
// expected-error@*:* 0+ {{multiple overloads}}
// expected-error@*:* 0+ {{cannot initialize a parameter of type 'const void *'}}
// expected-error@*:* 1+ {{'std::allocator' cannot allocate}}
// expected-error@*:* 1+ {{cannot form a reference to 'void'}}
// expected-error@*:* 1+ {{declared as a pointer}}
// expected-error@*:* 1+ {{implicit instantiation of undefined template}}
// expected-error@*:* 1+ {{must be standard-layout}}
// expected-error@*:* 1+ {{no function template}}
// expected-error@*:* 1+ {{no matching function}}
// expected-error@*:* 1+ {{no member named 'rebind'}}
// expected-warning@*:* 0+ {{volatile-qualified parameter type}}
