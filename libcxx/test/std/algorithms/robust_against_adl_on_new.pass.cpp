//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

#include <algorithm>

#include "test_macros.h"

struct A {
    int i;
    A(int v) : i(v) {}
    bool operator<(const A& rhs) const { return i < rhs.i; }
    static bool isEven(const A& a) { return a.i % 2 == 0; }
};

void *operator new(size_t, A*) = delete;

int main(int, char**)
{
    A a[4] = { 1,2,3,4 };
    std::sort(a, a+4);
    std::partition(a, a+4, A::isEven);
    std::stable_sort(a, a+4);
    std::stable_partition(a, a+4, A::isEven);

    return 0;
}
