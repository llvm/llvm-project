//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

#include <cstddef>

#include "test_macros.h"

#ifndef offsetof
#error offsetof not defined
#endif

struct A
{
    int x;
};

int main()
{
    static_assert(noexcept(offsetof(A, x)), "");
}
