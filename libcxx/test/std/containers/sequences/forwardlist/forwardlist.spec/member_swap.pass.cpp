//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void swap(forward_list& x);

#include <forward_list>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef test_allocator<T> A;
        typedef std::forward_list<T, A> C;
        const T t1[] = {0, 1, 2, 3, 4, 5};
        C c1(std::begin(t1), std::end(t1), A(1, 1));
        const T t2[] = {10, 11, 12};
        C c2(std::begin(t2), std::end(t2), A(1, 2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 3);
        assert(*std::next(c1.begin(), 0) == 10);
        assert(*std::next(c1.begin(), 1) == 11);
        assert(*std::next(c1.begin(), 2) == 12);
        assert(c1.get_allocator().get_id() == 1);

        assert(std::distance(c2.begin(), c2.end()) == 6);
        assert(*std::next(c2.begin(), 0) == 0);
        assert(*std::next(c2.begin(), 1) == 1);
        assert(*std::next(c2.begin(), 2) == 2);
        assert(*std::next(c2.begin(), 3) == 3);
        assert(*std::next(c2.begin(), 4) == 4);
        assert(*std::next(c2.begin(), 5) == 5);
        assert(c2.get_allocator().get_id() == 2);
    }
    {
        typedef int T;
        typedef test_allocator<T> A;
        typedef std::forward_list<T, A> C;
        const T t1[] = {0, 1, 2, 3, 4, 5};
        C c1(std::begin(t1), std::end(t1), A(1, 1));
        C c2(A(1, 2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 0);
        assert(c1.get_allocator().get_id() == 1);

        assert(std::distance(c2.begin(), c2.end()) == 6);
        assert(*std::next(c2.begin(), 0) == 0);
        assert(*std::next(c2.begin(), 1) == 1);
        assert(*std::next(c2.begin(), 2) == 2);
        assert(*std::next(c2.begin(), 3) == 3);
        assert(*std::next(c2.begin(), 4) == 4);
        assert(*std::next(c2.begin(), 5) == 5);
        assert(c2.get_allocator().get_id() == 2);
    }
    {
        typedef int T;
        typedef test_allocator<T> A;
        typedef std::forward_list<T, A> C;
        C c1(A(1, 1));
        const T t2[] = {10, 11, 12};
        C c2(std::begin(t2), std::end(t2), A(1, 2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 3);
        assert(*std::next(c1.begin(), 0) == 10);
        assert(*std::next(c1.begin(), 1) == 11);
        assert(*std::next(c1.begin(), 2) == 12);
        assert(c1.get_allocator().get_id() == 1);

        assert(std::distance(c2.begin(), c2.end()) == 0);
        assert(c2.get_allocator().get_id() == 2);
    }
    {
        typedef int T;
        typedef test_allocator<T> A;
        typedef std::forward_list<T, A> C;
        C c1(A(1, 1));
        C c2(A(1, 2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 0);
        assert(c1.get_allocator().get_id() == 1);

        assert(std::distance(c2.begin(), c2.end()) == 0);
        assert(c2.get_allocator().get_id() == 2);
    }

    {
        typedef int T;
        typedef other_allocator<T> A;
        typedef std::forward_list<T, A> C;
        const T t1[] = {0, 1, 2, 3, 4, 5};
        C c1(std::begin(t1), std::end(t1), A(1));
        const T t2[] = {10, 11, 12};
        C c2(std::begin(t2), std::end(t2), A(2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 3);
        assert(*std::next(c1.begin(), 0) == 10);
        assert(*std::next(c1.begin(), 1) == 11);
        assert(*std::next(c1.begin(), 2) == 12);
        assert(c1.get_allocator() == A(2));

        assert(std::distance(c2.begin(), c2.end()) == 6);
        assert(*std::next(c2.begin(), 0) == 0);
        assert(*std::next(c2.begin(), 1) == 1);
        assert(*std::next(c2.begin(), 2) == 2);
        assert(*std::next(c2.begin(), 3) == 3);
        assert(*std::next(c2.begin(), 4) == 4);
        assert(*std::next(c2.begin(), 5) == 5);
        assert(c2.get_allocator() == A(1));
    }
    {
        typedef int T;
        typedef other_allocator<T> A;
        typedef std::forward_list<T, A> C;
        const T t1[] = {0, 1, 2, 3, 4, 5};
        C c1(std::begin(t1), std::end(t1), A(1));
        C c2(A(2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 0);
        assert(c1.get_allocator() == A(2));

        assert(std::distance(c2.begin(), c2.end()) == 6);
        assert(*std::next(c2.begin(), 0) == 0);
        assert(*std::next(c2.begin(), 1) == 1);
        assert(*std::next(c2.begin(), 2) == 2);
        assert(*std::next(c2.begin(), 3) == 3);
        assert(*std::next(c2.begin(), 4) == 4);
        assert(*std::next(c2.begin(), 5) == 5);
        assert(c2.get_allocator() == A(1));
    }
    {
        typedef int T;
        typedef other_allocator<T> A;
        typedef std::forward_list<T, A> C;
        C c1(A(1));
        const T t2[] = {10, 11, 12};
        C c2(std::begin(t2), std::end(t2), A(2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 3);
        assert(*std::next(c1.begin(), 0) == 10);
        assert(*std::next(c1.begin(), 1) == 11);
        assert(*std::next(c1.begin(), 2) == 12);
        assert(c1.get_allocator() == A(2));

        assert(std::distance(c2.begin(), c2.end()) == 0);
        assert(c2.get_allocator() == A(1));
    }
    {
        typedef int T;
        typedef other_allocator<T> A;
        typedef std::forward_list<T, A> C;
        C c1(A(1));
        C c2(A(2));
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 0);
        assert(c1.get_allocator() == A(2));

        assert(std::distance(c2.begin(), c2.end()) == 0);
        assert(c2.get_allocator() == A(1));
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef min_allocator<T> A;
        typedef std::forward_list<T, A> C;
        const T t1[] = {0, 1, 2, 3, 4, 5};
        C c1(std::begin(t1), std::end(t1), A());
        const T t2[] = {10, 11, 12};
        C c2(std::begin(t2), std::end(t2), A());
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 3);
        assert(*std::next(c1.begin(), 0) == 10);
        assert(*std::next(c1.begin(), 1) == 11);
        assert(*std::next(c1.begin(), 2) == 12);
        assert(c1.get_allocator() == A());

        assert(std::distance(c2.begin(), c2.end()) == 6);
        assert(*std::next(c2.begin(), 0) == 0);
        assert(*std::next(c2.begin(), 1) == 1);
        assert(*std::next(c2.begin(), 2) == 2);
        assert(*std::next(c2.begin(), 3) == 3);
        assert(*std::next(c2.begin(), 4) == 4);
        assert(*std::next(c2.begin(), 5) == 5);
        assert(c2.get_allocator() == A());
    }
    {
        typedef int T;
        typedef min_allocator<T> A;
        typedef std::forward_list<T, A> C;
        const T t1[] = {0, 1, 2, 3, 4, 5};
        C c1(std::begin(t1), std::end(t1), A());
        C c2(A{});
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 0);
        assert(c1.get_allocator() == A());

        assert(std::distance(c2.begin(), c2.end()) == 6);
        assert(*std::next(c2.begin(), 0) == 0);
        assert(*std::next(c2.begin(), 1) == 1);
        assert(*std::next(c2.begin(), 2) == 2);
        assert(*std::next(c2.begin(), 3) == 3);
        assert(*std::next(c2.begin(), 4) == 4);
        assert(*std::next(c2.begin(), 5) == 5);
        assert(c2.get_allocator() == A());
    }
    {
        typedef int T;
        typedef min_allocator<T> A;
        typedef std::forward_list<T, A> C;
        C c1(A{});
        const T t2[] = {10, 11, 12};
        C c2(std::begin(t2), std::end(t2), A());
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 3);
        assert(*std::next(c1.begin(), 0) == 10);
        assert(*std::next(c1.begin(), 1) == 11);
        assert(*std::next(c1.begin(), 2) == 12);
        assert(c1.get_allocator() == A());

        assert(std::distance(c2.begin(), c2.end()) == 0);
        assert(c2.get_allocator() == A());
    }
    {
        typedef int T;
        typedef min_allocator<T> A;
        typedef std::forward_list<T, A> C;
        C c1(A{});
        C c2(A{});
        c1.swap(c2);

        assert(std::distance(c1.begin(), c1.end()) == 0);
        assert(c1.get_allocator() == A());

        assert(std::distance(c2.begin(), c2.end()) == 0);
        assert(c2.get_allocator() == A());
    }
#endif

  return 0;
}
