//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// deque()

#include <deque>
#include <cassert>

#include "test_allocator.h"
#include "../../../NotConstructible.h"
#include "min_allocator.h"

template <class T, class Allocator>
void
test()
{
    std::deque<T, Allocator> d;
    assert(d.size() == 0);
#if TEST_STD_VER >= 11
    std::deque<T, Allocator> d1 = {};
    assert(d1.size() == 0);
#endif
}

int main()
{
    test<int, std::allocator<int> >();
    test<NotConstructible, limited_allocator<NotConstructible, 1> >();
#if TEST_STD_VER >= 11
    test<int, min_allocator<int> >();
    test<NotConstructible, min_allocator<NotConstructible> >();
#endif
}
