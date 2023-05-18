//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// explicit forward_list(size_type n);
// explicit forward_list(size_type n, const Alloc& a);

#include <forward_list>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

template <class T, class Allocator>
void check_allocator(unsigned n, Allocator const &alloc = Allocator())
{
#if TEST_STD_VER > 11
    typedef std::forward_list<T, Allocator> C;
    C d(n, alloc);
    assert(d.get_allocator() == alloc);
    assert(static_cast<std::size_t>(std::distance(d.begin(), d.end())) == n);
#else
    ((void)n);
    ((void)alloc);
#endif
}

int main(int, char**)
{
    { // test that the ctor is explicit
      typedef std::forward_list<DefaultOnly> C;
      static_assert((std::is_constructible<C, std::size_t>::value), "");
      static_assert((!std::is_convertible<std::size_t, C>::value), "");
    }
    {
        typedef DefaultOnly T;
        typedef std::forward_list<T> C;
        unsigned N = 10;
        C c(N);
        unsigned n = 0;

        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n) {
#if TEST_STD_VER >= 11
            assert(*i == T());
#else
            ((void)0);
#endif
        }
        assert(n == N);
    }
#if TEST_STD_VER >= 11
    {
        typedef DefaultOnly T;
        typedef std::forward_list<T, min_allocator<T>> C;
        unsigned N = 10;
        C c(N);
        unsigned n = 0;
        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n)
            assert(*i == T());
        assert(n == N);
        check_allocator<T, min_allocator<T>> ( 0 );
        check_allocator<T, min_allocator<T>> ( 3 );
    }
#endif

  return 0;
}
