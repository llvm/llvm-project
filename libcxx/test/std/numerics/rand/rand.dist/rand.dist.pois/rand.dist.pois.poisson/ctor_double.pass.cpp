//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class poisson_distribution

// explicit poisson_distribution(double mean = 1.0);     // before C++20
// poisson_distribution() : poisson_distribution(1.0) {} // C++20
// explicit poisson_distribution(double mean);           // C++20

#include <random>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "make_implicit.h"
#include "test_convertible.h"
#endif

template <class T>
void test_implicit() {
#if TEST_STD_VER >= 11
  typedef std::poisson_distribution<T> D;
  static_assert(test_convertible<D>(), "");
  assert(D(1) == make_implicit<D>());
  static_assert(!test_convertible<D, double>(), "");
#endif
}

int main(int, char**)
{
    {
        typedef std::poisson_distribution<> D;
        D d;
        assert(d.mean() == 1);
    }
    {
        typedef std::poisson_distribution<> D;
        D d(3.5);
        assert(d.mean() == 3.5);
    }

    test_implicit<int>();
    test_implicit<long>();

    return 0;
}
