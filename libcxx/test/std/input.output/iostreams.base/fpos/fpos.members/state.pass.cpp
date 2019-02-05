//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class StateT> class fpos

// void state(stateT s);

#include <ios>
#include <cassert>

int main()
{
    std::fpos<int> f;
    f.state(3);
    assert(f.state() == 3);
}
