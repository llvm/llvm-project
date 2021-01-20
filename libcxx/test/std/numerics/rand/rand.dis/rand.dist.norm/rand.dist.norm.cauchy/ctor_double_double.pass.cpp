//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class cauchy_distribution

// explicit cauchy_distribution(RealType a = 0.0, RealType b = 1.0); // before C++20
// cauchy_distribution() : cauchy_distribution(0.0) {}               // C++20
// explicit cauchy_distribution(RealType a, RealType b = 1.0);       // C++20

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
  typedef std::cauchy_distribution<T> D;
  static_assert(test_convertible<D>(), "");
  assert(D(0) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
  static_assert(!test_convertible<D, T, T>(), "");
#endif
}

int main(int, char**)
{
    {
        typedef std::cauchy_distribution<> D;
        D d;
        assert(d.a() == 0);
        assert(d.b() == 1);
    }
    {
        typedef std::cauchy_distribution<> D;
        D d(14.5);
        assert(d.a() == 14.5);
        assert(d.b() == 1);
    }
    {
        typedef std::cauchy_distribution<> D;
        D d(14.5, 5.25);
        assert(d.a() == 14.5);
        assert(d.b() == 5.25);
    }

    test_implicit<float>();
    test_implicit<double>();

    return 0;
}
