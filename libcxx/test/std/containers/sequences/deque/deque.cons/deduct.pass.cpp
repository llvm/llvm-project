//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>
// UNSUPPORTED: c++03, c++11, c++14

// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    deque(InputIterator, InputIterator, Allocator = Allocator())
//    -> deque<typename iterator_traits<InputIterator>::value_type, Allocator>;
//
// template<ranges::input_range R, class Allocator = allocator<ranges::range_value_t<R>>>
//   deque(from_range_t, R&&, Allocator = Allocator())
//     -> deque<ranges::range_value_t<R>, Allocator>; // C++23

#include "asan_testing.h"
#include <array>
#include <cassert>
#include <climits> // INT_MAX
#include <cstddef>
#include <deque>
#include <iterator>

#include "deduction_guides_sfinae_checks.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

struct A {};

int main(int, char**)
{

//  Test the explicit deduction guides
    {
    const int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::deque deq(std::begin(arr), std::end(arr));

    static_assert(std::is_same_v<decltype(deq), std::deque<int>>, "");
    assert(std::equal(deq.begin(), deq.end(), std::begin(arr), std::end(arr)));
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

    {
    const long arr[] = {INT_MAX, 1L, 2L, 3L };
    std::deque deq(std::begin(arr), std::end(arr), std::allocator<long>());
    static_assert(std::is_same_v<decltype(deq)::value_type, long>, "");
    assert(deq.size() == 4);
    assert(deq[0] == INT_MAX);
    assert(deq[1] == 1L);
    assert(deq[2] == 2L);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

//  Test the implicit deduction guides

    {
//  We don't expect this one to work.
//  std::deque deq(std::allocator<int>()); // deque (allocator &)
    }

    {
    std::deque deq(1, A{}); // deque (size_type, T)
    static_assert(std::is_same_v<decltype(deq)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(deq)::allocator_type, std::allocator<A>>, "");
    assert(deq.size() == 1);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

    {
    std::deque deq(1, A{}, test_allocator<A>()); // deque (size_type, T, allocator)
    static_assert(std::is_same_v<decltype(deq)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(deq)::allocator_type, test_allocator<A>>, "");
    assert(deq.size() == 1);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

    {
    std::deque deq{1U, 2U, 3U, 4U, 5U}; // deque(initializer-list)
    static_assert(std::is_same_v<decltype(deq)::value_type, unsigned>, "");
    assert(deq.size() == 5);
    assert(deq[2] == 3U);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

    {
    std::deque deq({1.0, 2.0, 3.0, 4.0}, test_allocator<double>()); // deque(initializer-list, allocator)
    static_assert(std::is_same_v<decltype(deq)::value_type, double>, "");
    static_assert(std::is_same_v<decltype(deq)::allocator_type, test_allocator<double>>, "");
    assert(deq.size() == 4);
    assert(deq[3] == 4.0);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

    {
    std::deque<long double> source;
    std::deque deq(source); // deque(deque &)
    static_assert(std::is_same_v<decltype(deq)::value_type, long double>, "");
    static_assert(std::is_same_v<decltype(deq)::allocator_type, std::allocator<long double>>, "");
    assert(deq.size() == 0);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
    }

    {
        typedef test_allocator<short> Alloc;
        typedef test_allocator<int> ConvertibleToAlloc;

        {
        std::deque<short, Alloc> source;
        std::deque deq(source, Alloc(2));
        static_assert(std::is_same_v<decltype(deq), decltype(source)>);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(source));
        }

        {
        std::deque<short, Alloc> source;
        std::deque deq(source, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(deq), decltype(source)>);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(source));
        }

        {
        std::deque<short, Alloc> source;
        std::deque deq(std::move(source), Alloc(2));
        static_assert(std::is_same_v<decltype(deq), decltype(source)>);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(source));
        }

        {
        std::deque<short, Alloc> source;
        std::deque deq(std::move(source), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(deq), decltype(source)>);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(deq));
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(source));
        }
    }

#if TEST_STD_VER >= 23
    {
      {
        std::deque c(std::from_range, std::array<int, 0>());
        static_assert(std::is_same_v<decltype(c), std::deque<int>>);
      }

      {
        using Alloc = test_allocator<int>;
        std::deque c(std::from_range, std::array<int, 0>(), Alloc());
        static_assert(std::is_same_v<decltype(c), std::deque<int, Alloc>>);
      }
    }
#endif

    SequenceContainerDeductionGuidesSfinaeAway<std::deque, std::deque<int>>();

    return 0;
}
