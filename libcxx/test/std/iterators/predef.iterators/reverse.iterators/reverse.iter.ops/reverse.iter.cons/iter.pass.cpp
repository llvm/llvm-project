//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// explicit constexpr reverse_iterator(Iter x);
//
// constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
void
test(It i)
{
    std::reverse_iterator<It> r(i);
    assert(r.base() == i);
}

int main(int, char**)
{
    const char s[] = "123";
    test(bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s));
    test(s);

#if TEST_STD_VER > 14
    {
        constexpr const char *p = "123456789";
        constexpr std::reverse_iterator<const char *> it(p);
        static_assert(it.base() == p);
    }
#endif

  return 0;
}
