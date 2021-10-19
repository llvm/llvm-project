//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last,
//                  const Compare& comp, Container&& c, const Alloc& a);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "MoveOnly.h"

template<class T, class Cont, class Comp = std::less<T> >
struct PQ : std::priority_queue<T, Cont, Comp> {
    typedef std::priority_queue<T, Cont, Comp> base;

    template<class It, class Alloc>
    explicit PQ(It first, It last, const Comp& compare, Cont&& v, const Alloc& a) : base(first, last, compare, std::move(v), a) {}

    using base::c;
};

int main(int, char**)
{
    using Alloc = test_allocator<MoveOnly>;
    int a[] = {3, 5, 2, 0, 6, 8, 1};
    PQ<MoveOnly, std::vector<MoveOnly, Alloc>> q(
        a+3, a+7, std::less<MoveOnly>(),
        std::vector<MoveOnly, Alloc>(a, a+3), Alloc(2));
    assert(q.size() == 7);
    assert(q.top() == MoveOnly(8));
    assert(q.c.get_allocator() == Alloc(2));

    return 0;
}
