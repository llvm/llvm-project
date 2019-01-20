//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// XFAIL: c++11, c++14

// <functional>

#include <functional>
#include <type_traits>

#include "test_macros.h"

struct S : public std::function<void()> { using function::function; };

int main() {
   S f1( [](){} );
   S f2(std::allocator_arg, std::allocator<int>{}, f1);
}
