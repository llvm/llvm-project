//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// vector(size_type n, const value_type& x);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class C>
void
test(typename C::size_type n, const typename C::value_type& x)
{
    C c(n, x);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == n);
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
        assert(*i == x);
}

int main(int, char**)
{
    test<std::vector<bool> >(50, true);
#if TEST_STD_VER >= 11
    test<std::vector<bool, min_allocator<bool>> >(50, true);
#endif

  return 0;
}
