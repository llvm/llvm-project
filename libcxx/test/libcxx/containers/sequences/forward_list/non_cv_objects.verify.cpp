//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form forward_lists of object types.

#include <forward_list>

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
std::forward_list<int[2]> C10;
// expected-error@*:* 2 {{'std::forward_list' cannot hold C arrays}}
