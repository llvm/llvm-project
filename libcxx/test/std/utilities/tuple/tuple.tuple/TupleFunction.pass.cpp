//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// This is for bugs 18853 and 19118

#include <tuple>
#include <functional>

#include "test_macros.h"

struct X
{
    X() {}

    template <class T>
    X(T);

    void operator()() {}
};

int main(int, char**)
{
    X x;
    std::function<void()> f(x);

  return 0;
}
