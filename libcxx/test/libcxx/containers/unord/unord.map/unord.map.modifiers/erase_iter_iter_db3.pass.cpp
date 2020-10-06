//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Call erase(const_iterator first, const_iterator last); with both iterators from another container

// This test requires debug mode, which the library on macOS doesn't have.
// UNSUPPORTED: with_system_cxx_lib=macosx

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_map>
#include <cassert>
#include <exception>
#include <cstdlib>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::pair<int, int> P;
    P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
    std::unordered_map<int, int> l1(a1, a1+3);
    std::unordered_map<int, int> l2(a1, a1+3);
    std::unordered_map<int, int>::iterator i = l1.erase(l2.cbegin(), next(l2.cbegin()));
    assert(false);
    }
}
