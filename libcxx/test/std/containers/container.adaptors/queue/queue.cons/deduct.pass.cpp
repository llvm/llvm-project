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
//
// template<ranges::input_range R>
//   queue(from_range_t, R&&) -> queue<ranges::range_value_t<R>>; // since C++23
//
// template<ranges::input_range R, class Allocator>
//     queue(from_range_t, R&&, Allocator)
//       -> queue<ranges::range_value_t<R>, deque<ranges::range_value_t<R>, Allocator>>; // since C++23

#include <array>
#include <queue>
#include <list>
#include <iterator>
#include <cassert>
#include <cstddef>

#include "deduction_guides_sfinae_checks.h"
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

#if TEST_STD_VER >= 23
    {
        typedef short T;
        typedef test_allocator<T> Alloc;
        std::list<T> a;
        {
        std::queue q(a.begin(), a.end());
        static_assert(std::is_same_v<decltype(q), std::queue<T>>);
        }
        {
        std::queue q(a.begin(), a.end(), Alloc());
        static_assert(std::is_same_v<decltype(q), std::queue<T, std::deque<T, Alloc>>>);
        }
    }

    {
      {
        std::queue c(std::from_range, std::array<int, 0>());
        static_assert(std::is_same_v<decltype(c), std::queue<int>>);
      }

      {
        using Alloc = test_allocator<int>;
        std::queue c(std::from_range, std::array<int, 0>(), Alloc());
        static_assert(std::is_same_v<decltype(c), std::queue<int, std::deque<int, Alloc>>>);
      }
    }
#endif

    ContainerAdaptorDeductionGuidesSfinaeAway<std::queue, std::queue<int>>();

    return 0;
}
