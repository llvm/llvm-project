//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// explicit stack(const container_type& c);

#include <stack>
#include <cassert>
#include <cstddef>

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push_back(i);
    return c;
}

int main()
{
    std::deque<int> d = make<std::deque<int> >(5);
    std::stack<int> q(d);
    assert(q.size() == 5);
    for (std::size_t i = 0; i < d.size(); ++i)
    {
        assert(q.top() == d[d.size() - i - 1]);
        q.pop();
    }
}
