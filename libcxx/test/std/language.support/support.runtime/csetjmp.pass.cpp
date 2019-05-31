//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <csetjmp>

#include <csetjmp>
#include <type_traits>

#include "test_macros.h"

#ifndef setjmp
#error setjmp not defined
#endif

int main(int, char**)
{
    std::jmp_buf jb;
    ((void)jb); // Prevent unused warning
    static_assert((std::is_same<decltype(std::longjmp(jb, 0)), void>::value),
                  "std::is_same<decltype(std::longjmp(jb, 0)), void>::value");

  return 0;
}
