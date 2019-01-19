//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last,
//                  const Compare& comp, const container_type& c);

#include <queue>
#include <cassert>

int main()
{
    int a[] = {3, 5, 2, 0, 6, 8, 1};
    const int n = sizeof(a)/sizeof(a[0]);
    std::vector<int> v(a, a+n/2);
    std::priority_queue<int> q(a+n/2, a+n, std::less<int>(), v);
    assert(q.size() == n);
    assert(q.top() == 8);
}
