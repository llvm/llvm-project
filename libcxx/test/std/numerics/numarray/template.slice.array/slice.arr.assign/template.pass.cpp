//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template <class T> class slice_array

// void operator=(const T& value) const;

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    double a[] = { 0, 0, 0 };
    std::valarray<double> m(a, sizeof(a)/sizeof(a[0]));
    std::slice_array<double> s = m[std::slice(0, 3, 1)];
    s = 42;
    assert(m[0] == 42);
    assert(m[1] == 42);
    assert(m[2] == 42);

    ASSERT_SAME_TYPE(decltype(s = 42), void);

    return 0;
}
