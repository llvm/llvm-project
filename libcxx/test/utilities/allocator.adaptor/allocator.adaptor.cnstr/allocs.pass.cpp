//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
//   class scoped_allocator_adaptor

// template <class OuterA2>
//   scoped_allocator_adaptor(OuterA2&& outerAlloc,
//                            const InnerAllocs& ...innerAllocs);

#include <scoped_allocator>
#include <cassert>

#include "../allocators.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A1<int> a3(3);
        A a(a3);
        assert(a.outer_allocator() == A1<int>(3));
        assert(a.inner_allocator() == a);
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
    }
    A1<int>::copy_called = false;
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a(A1<int>(3));
        assert(a.outer_allocator() == A1<int>(3));
        assert(a.inner_allocator() == a);
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == true);
    }
    A1<int>::move_called = false;
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A1<int> a4(4);
        A a(a4, A2<int>(5));
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(a.outer_allocator() == A1<int>(4));
        assert(a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>>(A2<int>(5)));
    }
    A1<int>::copy_called = false;
    A1<int>::move_called = false;
    A2<int>::copy_called = false;
    A2<int>::move_called = false;
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a(A1<int>(4), A2<int>(5));
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == true);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(a.outer_allocator() == A1<int>(4));
        assert(a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>>(A2<int>(5)));
    }
    A1<int>::copy_called = false;
    A1<int>::move_called = false;
    A2<int>::copy_called = false;
    A2<int>::move_called = false;
    A1<int>::move_called = false;
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A1<int> a4(4);
        A a(a4, A2<int>(5), A3<int>(6));
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(A3<int>::copy_called == true);
        assert(A3<int>::move_called == false);
        assert(a.outer_allocator() == A1<int>(4));
        assert((a.inner_allocator() ==
            std::scoped_allocator_adaptor<A2<int>, A3<int>>(A2<int>(5), A3<int>(6))));
    }
    A1<int>::copy_called = false;
    A1<int>::move_called = false;
    A2<int>::copy_called = false;
    A2<int>::move_called = false;
    A3<int>::copy_called = false;
    A3<int>::move_called = false;
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a(A1<int>(4), A2<int>(5), A3<int>(6));
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == true);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(A3<int>::copy_called == true);
        assert(A3<int>::move_called == false);
        assert(a.outer_allocator() == A1<int>(4));
        assert((a.inner_allocator() ==
            std::scoped_allocator_adaptor<A2<int>, A3<int>>(A2<int>(5), A3<int>(6))));
    }

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
