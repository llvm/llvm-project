//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T, class Alloc> struct uses_allocator;

#include <memory>
#include <vector>

#include "test_macros.h"

struct A
{
};

struct B
{
    typedef int allocator_type;
};

struct C {
  static int allocator_type;
};

struct D {
  static int allocator_type() { return 0; }
};

struct E {
private:
  typedef int allocator_type;
};

template <bool Expected, class T, class A>
void
test()
{
    static_assert((std::uses_allocator<T, A>::value == Expected), "");
#if TEST_STD_VER > 14
    static_assert((std::uses_allocator_v<T, A> == Expected), "");
#endif
}

int main(int, char**)
{
    test<false, int, std::allocator<int> >();
    test<true, std::vector<int>, std::allocator<int> >();
    test<false, A, std::allocator<int> >();
    test<false, B, std::allocator<int> >();
    test<true, B, double>();
    test<false, C, decltype(C::allocator_type)>();
    test<false, D, decltype(D::allocator_type)>();
#if TEST_STD_VER >= 11
    test<false, E, int>();
#endif


//     static_assert((!std::uses_allocator<int, std::allocator<int> >::value), "");
//     static_assert(( std::uses_allocator<std::vector<int>, std::allocator<int> >::value), "");
//     static_assert((!std::uses_allocator<A, std::allocator<int> >::value), "");
//     static_assert((!std::uses_allocator<B, std::allocator<int> >::value), "");
//     static_assert(( std::uses_allocator<B, double>::value), "");
//     static_assert((!std::uses_allocator<C, decltype(C::allocator_type)>::value), "");
//     static_assert((!std::uses_allocator<D, decltype(D::allocator_type)>::value), "");
// #if TEST_STD_VER >= 11
//     static_assert((!std::uses_allocator<E, int>::value), "");
// #endif

  return 0;
}
