//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// http://wg21.link/LWG2447 gives implementors freedom to reject cv-qualified and non-object types in `std::allocator`.

#include <memory>
#include "test_macros.h"

std::allocator<const int> A1;
// expected-error@*:* {{'std::allocator' cannot allocate const types}}

std::allocator<volatile int> A2;
// expected-error@*:* {{'std::allocator' cannot allocate volatile types}}

std::allocator<int&> A3;
std::allocator<int&&> A4;
// expected-error@*:* 2 {{'std::allocator' cannot allocate references}}

std::allocator<int()> A5;
std::allocator<int(int)> A6;
std::allocator<int(int, int)> A7;
// expected-error@*:* 3 {{'std::allocator' cannot allocate functions}}

// Spurious errors
// expected-error@*:* 0+ {{multiple overloads}}
// expected-error@*:* 1+ {{declared as a pointer to a reference}}
