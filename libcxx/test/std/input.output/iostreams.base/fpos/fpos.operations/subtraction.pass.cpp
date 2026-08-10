//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class StateT> class fpos

// Subtraction with offset

#include <ios>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::fpos<std::mbstate_t> P;
    P p(11);
    std::streamoff o(6);
    P q = p - o;
    assert(q == P(5));
    p -= o;
    assert(p == q);

  return 0;
}
