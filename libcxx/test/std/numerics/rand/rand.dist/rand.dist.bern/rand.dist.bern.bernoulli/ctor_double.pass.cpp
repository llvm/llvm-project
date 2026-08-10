//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// explicit bernoulli_distribution(double p = 0.5);          // before C++20
// bernoulli_distribution() : bernoulli_distribution(0.5) {} // C++20
// explicit bernoulli_distribution(double p);                // C++20

#include <random>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "make_implicit.h"
#include "test_convertible.h"
#endif

int main(int, char**)
{
    {
        typedef std::bernoulli_distribution D;
        D d;
        assert(d.p() == 0.5);
    }
    {
        typedef std::bernoulli_distribution D;
        D d(0);
        assert(d.p() == 0);
    }
    {
        typedef std::bernoulli_distribution D;
        D d(0.75);
        assert(d.p() == 0.75);
    }

#if TEST_STD_VER >= 11
    {
      typedef std::bernoulli_distribution D;
      static_assert(test_convertible<D>(), "");
      assert(D(0.5) == make_implicit<D>());
      static_assert(!test_convertible<D, double>(), "");
    }
#endif

    return 0;
}
