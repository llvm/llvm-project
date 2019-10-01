//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// template <class Alloc>
//   explicit stack(const Alloc& a);

#include <stack>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

struct test
    : private std::stack<int, std::deque<int, test_allocator<int> > >
{
    typedef std::stack<int, std::deque<int, test_allocator<int> > > base;

    explicit test(const test_allocator<int>& a) : base(a) {}
    test(const container_type& c, const test_allocator<int>& a) : base(c, a) {}
#if TEST_STD_VER >= 11
    test(container_type&& c, const test_allocator<int>& a) : base(std::move(c), a) {}
    test(test&& q, const test_allocator<int>& a) : base(std::move(q), a) {}
#endif
    test_allocator<int> get_allocator() {return c.get_allocator();}
};

int main(int, char**)
{
    test q(test_allocator<int>(3));
    assert(q.get_allocator() == test_allocator<int>(3));

  return 0;
}
