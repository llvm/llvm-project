//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class binomial_distribution

// result_type max() const;

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::binomial_distribution<> D;
        D d(4, .25);
        assert(d.max() == 4);
    }

  return 0;
}
