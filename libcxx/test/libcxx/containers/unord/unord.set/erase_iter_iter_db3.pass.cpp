//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Call erase(const_iterator first, const_iterator last); with both iterators from another container

// This test requires debug mode, which the library on macOS doesn't have.
// UNSUPPORTED: with_system_cxx_lib=macosx

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_set>
#include <cassert>
#include <exception>
#include <cstdlib>

#include "test_macros.h"

int main(int, char**)
{
    {
    int a1[] = {1, 2, 3};
    std::unordered_set<int> l1(a1, a1+3);
    std::unordered_set<int> l2(a1, a1+3);
    std::unordered_set<int>::iterator i = l1.erase(l2.cbegin(), next(l2.cbegin()));
    assert(false);
    }
}
