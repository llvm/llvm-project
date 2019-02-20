//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// pointer operator->() const;
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"

template <class It>
void
test(It i)
{
    std::move_iterator<It> r(i);
    assert(r.operator->() == i);
}

int main(int, char**)
{
    char s[] = "123";
    test(s);

#if TEST_STD_VER > 14
    {
    constexpr const char *p = "123456789";
    typedef std::move_iterator<const char *> MI;
    constexpr MI it1 = std::make_move_iterator(p);
    constexpr MI it2 = std::make_move_iterator(p+1);
    static_assert(it1.operator->() == p, "");
    static_assert(it2.operator->() == p + 1, "");
    }
#endif

  return 0;
}
