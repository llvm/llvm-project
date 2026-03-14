//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class lognormal_distribution

// explicit lognormal_distribution(RealType mean = 0.0, RealType stddev = 1.0); // before C++20
// lognormal_distribution() : lognormal_distribution(0.0) {}                    // C++20
// explicit lognormal_distribution(RealType mean, RealType stddev = 1.0);       // C++20

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
  typedef std::lognormal_distribution<T> D;
  static_assert(test_convertible<D>(), "");
  assert(D(0) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
  static_assert(!test_convertible<D, T, T>(), "");
#endif
}

int main(int, char**)
{
    {
        typedef std::lognormal_distribution<> D;
        D d;
        assert(d.m() == 0);
        assert(d.s() == 1);
    }
    {
        typedef std::lognormal_distribution<> D;
        D d(14.5);
        assert(d.m() == 14.5);
        assert(d.s() == 1);
    }
    {
        typedef std::lognormal_distribution<> D;
        D d(14.5, 5.25);
        assert(d.m() == 14.5);
        assert(d.s() == 5.25);
    }

    test_implicit<float>();
    test_implicit<double>();

    return 0;
}
