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

std::allocator<const int> A1;    // expected-error@*:* {{'std::allocator' can only allocate non-const object types}}
std::allocator<volatile int> A2; // expected-error@*:* {{'std::allocator' can only allocate non-volatile object types}}
std::allocator<int&>
    A3; // expected-error@*:*{{'std::allocator' can only allocate object types; references are not objects}}
std::allocator<int&&>
    A4; // expected-error@*:*{{'std::allocator' can only allocate object types; references are not objects}}
std::allocator<int()>
    A5; // expected-error@*:*{{'std::allocator' can only allocate object types; functions are not objects (consider using a function pointer)}}
std::allocator<int(int)>
    A6; // expected-error@*:*{{'std::allocator' can only allocate object types; functions are not objects (consider using a function pointer)}}
std::allocator<int(int, int)>
    A7; // expected-error@*:*{{'std::allocator' can only allocate object types; functions are not objects (consider using a function pointer)}}
