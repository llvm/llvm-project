//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// explicit vector(size_type n);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

template <class C>
void
test2(typename C::size_type n,
      typename C::allocator_type const& a = typename C::allocator_type ())
{
#if TEST_STD_VER >= 14
    C c(n, a);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == n);
    assert(c.get_allocator() == a);
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
        assert(*i == typename C::value_type());
#else
  ((void)n);
  ((void)a);
#endif
}

template <class C>
void
test1(typename C::size_type n)
{
    C c(n);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == n);
    assert(c.get_allocator() == typename C::allocator_type());
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
        assert(*i == typename C::value_type());
}

template <class C>
void
test(typename C::size_type n)
{
    test1<C> ( n );
    test2<C> ( n );
}

int main(int, char**)
{
    test<std::vector<bool> >(50);
#if TEST_STD_VER >= 11
    test<std::vector<bool, min_allocator<bool>> >(50);
    test2<std::vector<bool, test_allocator<bool>> >( 100, test_allocator<bool>(23));
#endif

  return 0;
}
