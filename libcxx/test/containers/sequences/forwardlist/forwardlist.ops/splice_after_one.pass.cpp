//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void splice_after(const_iterator p, forward_list&& x, const_iterator i);

#include <forward_list>
#include <cassert>
#include <iterator>

typedef int T;
typedef std::forward_list<T> C;
const T t1[] = {0, 1, 2, 3, 4, 5, 6, 7};
const T t2[] = {10, 11, 12};
const int size_t1 = std::end(t1) - std::begin(t1);
const int size_t2 = std::end(t2) - std::begin(t2);

void
testd(const C& c, int p, int f)
{
    C::const_iterator i = c.begin();
    int n1 = 0;
    for (; n1 < p; ++n1, ++i)
        assert(*i == t1[n1]);
    for (int n2 = f; n2 < f+1; ++n2, ++i)
        assert(*i == t2[n2]);
    for (; n1 < size_t1; ++n1, ++i)
        assert(*i == t1[n1]);
    assert(distance(c.begin(), c.end()) == size_t1 + 1);
}

void
tests(const C& c, int p, int f)
{
    C::const_iterator i = c.begin();
    int n = 0;
    int d = 1;
    if (p == f || p == f+1)
    {
        for (n = 0; n < size_t1; ++n, ++i)
            assert(*i == t1[n]);
    }
    else if (p < f)
    {
        for (n = 0; n < p; ++n, ++i)
            assert(*i == t1[n]);
        for (n = f; n < f+1; ++n, ++i)
            assert(*i == t1[n]);
        for (n = p; n < f; ++n, ++i)
            assert(*i == t1[n]);
        for (n = f+1; n < size_t1; ++n, ++i)
            assert(*i == t1[n]);
    }
    else // p > f+1
    {
        for (n = 0; n < f; ++n, ++i)
            assert(*i == t1[n]);
        for (n = f+1; n < p; ++n, ++i)
            assert(*i == t1[n]);
        for (n = f; n < f+1; ++n, ++i)
            assert(*i == t1[n]);
        for (n = p; n < size_t1; ++n, ++i)
            assert(*i == t1[n]);
    }
    assert(distance(c.begin(), c.end()) == size_t1);
}

int main()
{
    // splicing different containers
    for (int f = 0; f <= size_t2-1; ++f)
    {
        for (int p = 0; p <= size_t1; ++p)
        {
            C c1(std::begin(t1), std::end(t1));
            C c2(std::begin(t2), std::end(t2));

            c1.splice_after(next(c1.cbefore_begin(), p), std::move(c2),
                  next(c2.cbefore_begin(), f));
            testd(c1, p, f);
        }
    }

    // splicing within same container
    for (int f = 0; f <= size_t1-1; ++f)
    {
        for (int p = 0; p <= size_t1; ++p)
        {
            C c1(std::begin(t1), std::end(t1));

            c1.splice_after(next(c1.cbefore_begin(), p), std::move(c1),
                  next(c1.cbefore_begin(), f));
            tests(c1, p, f);
        }
    }
}
