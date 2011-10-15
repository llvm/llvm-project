//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// template <InputIterator Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

#include <list>
#include <cstdlib>
#include <cassert>

int throw_next = 0xFFFF;
int count = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    if (throw_next == 0)
        throw std::bad_alloc();
    --throw_next;
    ++count;
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    --count;
    std::free(p);
}

int main()
{
    int a1[] = {1, 2, 3};
    std::list<int> l1;
    std::list<int>::iterator i = l1.insert(l1.begin(), a1, a1+3);
    assert(i == l1.begin());
    assert(l1.size() == 3);
    assert(distance(l1.begin(), l1.end()) == 3);
    i = l1.begin();
    assert(*i == 1);
    ++i;
    assert(*i == 2);
    ++i;
    assert(*i == 3);
    int a2[] = {4, 5, 6};
    i = l1.insert(i, a2, a2+3);
    assert(*i == 4);
    assert(l1.size() == 6);
    assert(distance(l1.begin(), l1.end()) == 6);
    i = l1.begin();
    assert(*i == 1);
    ++i;
    assert(*i == 2);
    ++i;
    assert(*i == 4);
    ++i;
    assert(*i == 5);
    ++i;
    assert(*i == 6);
    ++i;
    assert(*i == 3);
    throw_next = 2;
    int save_count = count;
    try
    {
        i = l1.insert(i, a2, a2+3);
        assert(false);
    }
    catch (...)
    {
    }
    assert(save_count == count);
    assert(l1.size() == 6);
    assert(distance(l1.begin(), l1.end()) == 6);
    i = l1.begin();
    assert(*i == 1);
    ++i;
    assert(*i == 2);
    ++i;
    assert(*i == 4);
    ++i;
    assert(*i == 5);
    ++i;
    assert(*i == 6);
    ++i;
    assert(*i == 3);
}
