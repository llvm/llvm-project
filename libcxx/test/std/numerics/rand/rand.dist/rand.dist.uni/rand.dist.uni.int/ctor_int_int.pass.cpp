//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// explicit uniform_int_distribution(IntType a = 0,
//                                   IntType b = numeric_limits<IntType>::max()); // before C++20
// uniform_int_distribution() : uniform_int_distribution(0) {}                    // C++20
// explicit uniform_int_distribution(IntType a,
//                                   IntType b = numeric_limits<IntType>::max()); // C++20

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
  typedef std::uniform_int_distribution<> D;
  static_assert(test_convertible<D>(), "");
  assert(D(0) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
  static_assert(!test_convertible<D, T, T>(), "");
#endif
}

int main(int, char**)
{
    {
        typedef std::uniform_int_distribution<> D;
        D d;
        assert(d.a() == 0);
        assert(d.b() == std::numeric_limits<int>::max());
    }
    {
        typedef std::uniform_int_distribution<> D;
        D d(-6);
        assert(d.a() == -6);
        assert(d.b() == std::numeric_limits<int>::max());
    }
    {
        typedef std::uniform_int_distribution<> D;
        D d(-6, 106);
        assert(d.a() == -6);
        assert(d.b() == 106);
    }

    test_implicit<int>();
    test_implicit<long>();

    return 0;
}
