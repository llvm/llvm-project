//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is for bugs 18853 and 19118

#include "test_macros.h"

#if TEST_STD_VER >= 11

#include <tuple>
#include <functional>

struct X
{
    X() {}

    template <class T>
    X(T);

    void operator()() {}
};

int main()
{
    X x;
    std::function<void()> f(x);
}
#else
int main () {}
#endif
