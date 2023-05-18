//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// See https://llvm.org/PR20002

#include <functional>
#include <type_traits>

#include "test_macros.h"

using Fn = std::function<void()>;
struct S : public std::function<void()> { using function::function; };

int main(int, char**) {
    S s( [](){} );
    S f1( s );
#if TEST_STD_VER <= 14
    S f2(std::allocator_arg, std::allocator<int>{}, s);
#endif

  return 0;
}
