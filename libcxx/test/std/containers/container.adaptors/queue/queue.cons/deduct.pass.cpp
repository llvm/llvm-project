//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>
// UNSUPPORTED: c++03, c++11, c++14

// template<class Container>
//   queue(Container) -> queue<typename Container::value_type, Container>;
//
// template<class Container, class Allocator>
//   queue(Container, Allocator) -> queue<typename Container::value_type, Container>;


#include <queue>
#include <list>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <climits> // INT_MAX

#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

struct A {};

int main(int, char**)
{

//  Test the explicit deduction guides
    {
    std::list<int> l{0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::queue que(l);

    static_assert(std::is_same_v<decltype(que), std::queue<int, std::list<int>>>, "");
    assert(que.size() == l.size());
    assert(que.back() == l.back());
    }

    {
    std::list<long, test_allocator<long>> l{10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    std::queue que(l, test_allocator<long>(0,2)); // different allocator
    static_assert(std::is_same_v<decltype(que)::container_type, std::list<long, test_allocator<long>>>, "");
    static_assert(std::is_same_v<decltype(que)::value_type, long>, "");
    assert(que.size() == 10);
    assert(que.back() == 19);
//  I'd like to assert that we've gotten the right allocator in the queue, but
//  I don't know how to get at the underlying container.
    }

//  Test the implicit deduction guides
    {
//  We don't expect this one to work - no way to implicitly get value_type
//  std::queue que(std::allocator<int>()); // queue (allocator &)
    }

    {
    std::queue<A> source;
    std::queue que(source); // queue(queue &)
    static_assert(std::is_same_v<decltype(que)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(que)::container_type, std::deque<A>>, "");
    assert(que.size() == 0);
    }

    {
        typedef short T;
        typedef test_allocator<T> Alloc;
        typedef std::list<T, Alloc> Cont;
        typedef test_allocator<int> ConvertibleToAlloc;
        static_assert(std::uses_allocator_v<Cont, ConvertibleToAlloc> &&
                      !std::is_same_v<typename Cont::allocator_type, ConvertibleToAlloc>);

        {
        Cont cont;
        std::queue que(cont, Alloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }

        {
        Cont cont;
        std::queue que(cont, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }

        {
        Cont cont;
        std::queue que(std::move(cont), Alloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }

        {
        Cont cont;
        std::queue que(std::move(cont), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }
    }

    {
        typedef short T;
        typedef test_allocator<T> Alloc;
        typedef std::list<T, Alloc> Cont;
        typedef test_allocator<int> ConvertibleToAlloc;
        static_assert(std::uses_allocator_v<Cont, ConvertibleToAlloc> &&
                      !std::is_same_v<typename Cont::allocator_type, ConvertibleToAlloc>);

        {
        std::queue<T, Cont> source;
        std::queue que(source, Alloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }

        {
        std::queue<T, Cont> source;
        std::queue que(source, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }

        {
        std::queue<T, Cont> source;
        std::queue que(std::move(source), Alloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }

        {
        std::queue<T, Cont> source;
        std::queue que(std::move(source), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(que), std::queue<T, Cont>>);
        }
    }

    return 0;
}
