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
