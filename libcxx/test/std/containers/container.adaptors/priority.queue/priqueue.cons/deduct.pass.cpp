//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>
// UNSUPPORTED: c++03, c++11, c++14

// template<class Compare, class Container>
// priority_queue(Compare, Container)
//     -> priority_queue<typename Container::value_type, Container, Compare>;
//
// template<class InputIterator,
//          class Compare = less<typename iterator_traits<InputIterator>::value_type>,
//          class Container = vector<typename iterator_traits<InputIterator>::value_type>>
// priority_queue(InputIterator, InputIterator, Compare = Compare(), Container = Container())
//     -> priority_queue<typename iterator_traits<InputIterator>::value_type, Container, Compare>;
//
// template<class Compare, class Container, class Allocator>
// priority_queue(Compare, Container, Allocator)
//     -> priority_queue<typename Container::value_type, Container, Compare>;


#include <queue>
#include <vector>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <climits> // INT_MAX

#include "deduction_guides_sfinae_checks.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

struct A {};

int main(int, char**)
{

//  Test the explicit deduction guides
    {
    std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::priority_queue pri(std::greater<int>(), v); // priority_queue(Compare, Container)

    static_assert(std::is_same_v<decltype(pri), std::priority_queue<int, std::vector<int>, std::greater<int>>>, "");
    assert(pri.size() == v.size());
    assert(pri.top() == 0);
    }

    {
    std::vector<long, test_allocator<long>> v{10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    std::priority_queue pri(std::greater<long>(), v, test_allocator<long>(2)); // priority_queue(Compare, Container, Allocator)

    static_assert(std::is_same_v<decltype(pri),
                                 std::priority_queue<long, std::vector<long, test_allocator<long>>, std::greater<long>>>, "");
    assert(pri.size() == v.size());
    assert(pri.top() == 10);
    }

    {
    std::vector<short> v{10, 11, 12, 13, 14, 15, 28, 17, 18, 19 };
    std::priority_queue pri(v.begin(), v.end()); // priority_queue(Iter, Iter)

    static_assert(std::is_same_v<decltype(pri), std::priority_queue<short>>, "");
    assert(pri.size() == v.size());
    assert(pri.top() == 28);
    }

    {
    std::vector<double> v{10, 11, 12, 13, 6, 15, 28, 17, 18, 19 };
    std::priority_queue pri(v.begin(), v.end(), std::greater<double>()); // priority_queue(Iter, Iter, Comp)

    static_assert(std::is_same_v<decltype(pri), std::priority_queue<double, std::vector<double>, std::greater<double>>>, "");
    assert(pri.size() == v.size());
    assert(pri.top() == 6);
    }

    {
    std::vector<double> v{10, 6, 15, 28, 4, 18, 19 };
    std::deque<double> deq;
    std::priority_queue pri(v.begin(), v.end(), std::greater<double>(), deq); // priority_queue(Iter, Iter, Comp, Container)

    static_assert(std::is_same_v<decltype(pri), std::priority_queue<double, std::deque<double>, std::greater<double>>>, "");
    assert(pri.size() == v.size());
    assert(pri.top() == 4);
    }

//  Test the implicit deduction guides
    {
//  We don't expect this one to work - no way to implicitly get value_type
//  std::priority_queue pri(std::allocator<int>()); // queue (allocator &)
    }

    {
    std::priority_queue<float> source;
    std::priority_queue pri(source); // priority_queue(priority_queue &)
    static_assert(std::is_same_v<decltype(pri)::value_type, float>, "");
    static_assert(std::is_same_v<decltype(pri)::container_type, std::vector<float>>, "");
    assert(pri.size() == 0);
    }

    {
        typedef short T;
        typedef std::greater<T> Comp;
        typedef test_allocator<T> Alloc;
        typedef std::deque<T, Alloc> Cont;
        typedef test_allocator<int> ConvertibleToAlloc;
        static_assert(std::uses_allocator_v<Cont, ConvertibleToAlloc> &&
                      !std::is_same_v<typename Cont::allocator_type, ConvertibleToAlloc>);

        {
        Comp comp;
        Cont cont;
        std::priority_queue pri(comp, cont, Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Comp comp;
        Cont cont;
        std::priority_queue pri(comp, cont, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Comp comp;
        Cont cont;
        std::priority_queue pri(comp, std::move(cont), Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Comp comp;
        Cont cont;
        std::priority_queue pri(comp, std::move(cont), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }
    }

    {
        typedef short T;
        typedef signed char ConvertibleToT;
        typedef std::greater<T> Comp;
        typedef test_allocator<T> Alloc;
        typedef std::deque<T, Alloc> Cont;
        typedef test_allocator<int> ConvertibleToAlloc;
        static_assert(std::uses_allocator_v<Cont, ConvertibleToAlloc> &&
                      !std::is_same_v<typename Cont::allocator_type, ConvertibleToAlloc>);

        {
        std::priority_queue<T, Cont, Comp> source;
        std::priority_queue pri(source, Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        std::priority_queue<T, Cont, Comp> source;
        std::priority_queue pri(source, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        std::priority_queue<T, Cont, Comp> source;
        std::priority_queue pri(std::move(source), Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        std::priority_queue<T, Cont, Comp> source;
        std::priority_queue pri(std::move(source), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        std::priority_queue pri(Comp(), cont, Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        std::priority_queue pri(Comp(), cont, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        std::priority_queue pri(Comp(), std::move(cont), Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        std::priority_queue pri(Comp(), std::move(cont), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        T a[2] = {};
        std::priority_queue pri(a, a+2, Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, std::vector<T, Alloc>>>);
        }

        {
        T a[2] = {};
        std::priority_queue pri(a, a+2, Comp(), Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, std::vector<T, Alloc>, Comp>>);
        }

        {
        Cont cont;
        ConvertibleToT a[2] = {};
        std::priority_queue pri(a, a+2, Comp(), cont, Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        ConvertibleToT a[2] = {};
        std::priority_queue pri(a, a+2, Comp(), cont, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        ConvertibleToT a[2] = {};
        std::priority_queue pri(a, a+2, Comp(), std::move(cont), Alloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }

        {
        Cont cont;
        ConvertibleToT a[2] = {};
        std::priority_queue pri(a, a+2, Comp(), std::move(cont), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(pri), std::priority_queue<T, Cont, Comp>>);
        }
    }

    // Deduction guides should be SFINAE'd away when given:
    // - "bad" input iterators (that is, a type not qualifying as an input
    //   iterator);
    // - a bad allocator;
    // - an allocator instead of a comparator;
    // - an allocator instead of a container;
    // - an allocator and a container that uses a different allocator.
    {
        using Comp = std::less<int>;
        using Cont = std::vector<int>;
        using Alloc = std::allocator<int>;
        using Iter = int*;

        // The only requirement in the Standard is that integral types cannot be
        // considered input iterators, beyond that it is unspecified.
        using BadIter = int;
#ifdef _LIBCPP_VERSION
        struct OutputIter {
          using iterator_category = std::output_iterator_tag;
          using value_type = void;
          using difference_type = void;
          using pointer = void;
          using reference = void;

          const OutputIter& operator*() const { return *this; }
          const OutputIter& operator++() { return *this; }
          OutputIter operator++(int) const { return *this; }
        };
#endif // _LIBCPP_VERSION

        struct BadAlloc {};
        using AllocAsComp = Alloc;
        using AllocAsCont = Alloc;
        using DiffAlloc = test_allocator<int>;

        // (iter, iter)
        //
        // Cannot deduce from (BAD_iter, BAD_iter)
        static_assert(SFINAEs_away<std::priority_queue, BadIter, BadIter>);
        // Note: (OutputIter, OutputIter) is interpreted as (comp, cont) and fails on accessing
        // non-existent typedefs in `OutputIter` (as if it were a container). There is no
        // requirement to SFINAE away bad containers.

        // (iter, iter, comp)
        //
        // Cannot deduce from (BAD_iter, BAD_iter, comp)
        static_assert(SFINAEs_away<std::priority_queue, BadIter, BadIter, Comp>);
        LIBCPP_STATIC_ASSERT(SFINAEs_away<std::priority_queue, OutputIter, OutputIter, Comp>);
        // Note: (iter, iter, ALLOC_as_comp) is allowed -- it just calls (iter, iter, alloc).

        // (iter, iter, comp, cont)
        //
        // Cannot deduce from (BAD_iter, BAD_iter, comp, cont)
        static_assert(SFINAEs_away<std::priority_queue, BadIter, BadIter, Comp, Cont>);
        LIBCPP_STATIC_ASSERT(SFINAEs_away<std::priority_queue, OutputIter, OutputIter, Comp, Cont>);
        // Cannot deduce from (iter, iter, ALLOC_as_comp, cont)
        static_assert(SFINAEs_away<std::priority_queue, Iter, Iter, AllocAsComp, Cont>);
        // Note: (iter, iter, comp, ALLOC_as_cont) is allowed -- it just calls (iter, iter, comp,
        // alloc).

        // (iter, iter, alloc)
        //
        // Cannot deduce from (BAD_iter, BAD_iter, alloc)
        static_assert(SFINAEs_away<std::priority_queue, BadIter, BadIter, Alloc>);
        LIBCPP_STATIC_ASSERT(SFINAEs_away<std::priority_queue, OutputIter, OutputIter, Alloc>);
        // Note: (iter, iter, BAD_alloc) is interpreted as (iter, iter, comp) instead and fails upon
        // instantiation. There is no requirement to SFINAE away bad comparators.

        // (iter, iter, comp, alloc)
        //
        // Cannot deduce from (iter, iter, ALLOC_as_comp, alloc)
        static_assert(SFINAEs_away<std::priority_queue, Iter, Iter, AllocAsComp, Alloc>);
        // Note: (iter, iter, comp, BAD_alloc) is interpreted as (iter, iter, comp, cont) instead
        // and fails upon instantiation. There is no requirement to SFINAE away bad containers.

        // (iter, iter, comp, cont, alloc)
        //
        // Cannot deduce from (BAD_iter, BAD_iter, comp, cont, alloc)
        static_assert(SFINAEs_away<std::priority_queue, BadIter, BadIter, Comp, Cont, Alloc>);
        LIBCPP_STATIC_ASSERT(
            SFINAEs_away<std::priority_queue, OutputIter, OutputIter, Comp, Cont, Alloc>);
        // Cannot deduce from (iter, iter, ALLOC_as_comp, cont, alloc)
        static_assert(SFINAEs_away<std::priority_queue, Iter, Iter, AllocAsComp, Cont, Alloc>);
        // Cannot deduce from (iter, iter, comp, ALLOC_as_cont, alloc)
        static_assert(SFINAEs_away<std::priority_queue, Iter, Iter, Comp, AllocAsCont, Alloc>);
        // Cannot deduce from (iter, iter, comp, cont, BAD_alloc)
        static_assert(SFINAEs_away<std::priority_queue, Iter, Iter, Comp, Cont, BadAlloc>);
        // Cannot deduce from (iter, iter, comp, cont, DIFFERENT_alloc)
        static_assert(SFINAEs_away<std::priority_queue, Iter, Iter, Comp, Cont, DiffAlloc>);

        // (comp, alloc)
        //
        // Cannot deduce from (ALLOC_as_comp, alloc)
        static_assert(SFINAEs_away<std::priority_queue, AllocAsComp, Alloc>);
        // Cannot deduce from (comp, BAD_alloc)
        static_assert(SFINAEs_away<std::priority_queue, Comp, BadAlloc>);

        // (comp, cont, alloc)
        //
        // Cannot deduce from (ALLOC_as_comp, cont, alloc)
        static_assert(SFINAEs_away<std::priority_queue, AllocAsComp, Cont, Alloc>);
        // Cannot deduce from (comp, ALLOC_as_cont, alloc)
        static_assert(SFINAEs_away<std::priority_queue, Comp, AllocAsCont, Alloc>);
        // Cannot deduce from (comp, cont, BAD_alloc)
        static_assert(SFINAEs_away<std::priority_queue, Comp, Cont, BadAlloc>);
        // Cannot deduce from (comp, cont, DIFFERENT_alloc)
        static_assert(SFINAEs_away<std::priority_queue, Comp, Cont, DiffAlloc>);

        // (comp, cont)
        //
        // Cannot deduce from (ALLOC_as_comp, cont)
        static_assert(SFINAEs_away<std::priority_queue, AllocAsComp, Cont>);
        // Cannot deduce from (comp, ALLOC_as_cont)
        static_assert(SFINAEs_away<std::priority_queue, Comp, AllocAsCont>);
    }

    return 0;
}
