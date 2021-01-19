//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class chi_squared_distribution

// explicit chi_squared_distribution(RealType n = 1.0);          // before C++20
// chi_squared_distribution() : chi_squared_distribution(1.0) {} // C++20
// explicit chi_squared_distribution(RealType n);                // C++20

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
  typedef std::chi_squared_distribution<T> D;
  static_assert(test_convertible<D>(), "");
  assert(D(1) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
#endif
}

int main(int, char**)
{
    {
        typedef std::chi_squared_distribution<> D;
        D d;
        assert(d.n() == 1);
    }
    {
        typedef std::chi_squared_distribution<> D;
        D d(14.5);
        assert(d.n() == 14.5);
    }

    test_implicit<float>();
    test_implicit<double>();

    return 0;
}
