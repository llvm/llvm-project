//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// iterator erase(const_iterator f, const_iterator l)

//  Erasing items from the beginning or the end of a deque shall not invalidate iterators
//  to items that were not erased.


#include <deque>
#include <cstdint>
#include <cassert>

template <typename C>
void del_at_start(C c, size_t num)
{
    typename C::iterator first = c.begin();
    typename C::iterator last  = first + num;
    typename C::iterator it1 = last;
    typename C::iterator it2 = c.end() - 1;

    c.erase (first, last);

    typename C::iterator it3 = c.begin();
    typename C::iterator it4 = c.end() - 1;
    assert(  it1 ==   it3);
    assert( *it1 ==  *it3);
    assert(&*it1 == &*it3);
    assert(  it2 ==   it4);
    assert( *it2 ==  *it4);
    assert(&*it2 == &*it4);
}

template <typename C>
void del_at_end(C c, size_t num)
{
    typename C::iterator last  = c.end();
    typename C::iterator first = last - num;
    typename C::iterator it1 = c.begin();
    typename C::iterator it2 = first - 1;

    c.erase (first, last);

    typename C::iterator it3 = c.begin();
    typename C::iterator it4 = c.end() - 1;
    assert(  it1 ==   it3);
    assert( *it1 ==  *it3);
    assert(&*it1 == &*it3);
    assert(  it2 ==   it4);
    assert( *it2 ==  *it4);
    assert(&*it2 == &*it4);
}


int main()
{
    std::deque<int> queue;
    for (int i = 0; i < 20; ++i)
        queue.push_back(i);

    while (queue.size() > 1)
    {
        for (size_t i = 1; i < queue.size(); ++i)
        {
            del_at_start(queue, i);
            del_at_end  (queue, i);
        }
        queue.pop_back();
    }
}
