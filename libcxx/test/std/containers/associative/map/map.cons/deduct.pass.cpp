//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator,
//          class Compare = less<iter-value-type<InputIterator>>,
//          class Allocator = allocator<iter-value-type<InputIterator>>>
// map(InputIterator, InputIterator,
//          Compare = Compare(), Allocator = Allocator())
//   -> map<iter-value-type<InputIterator>, Compare, Allocator>;
// template<class Key, class Compare = less<Key>, class Allocator = allocator<Key>>
// map(initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
//   -> map<Key, Compare, Allocator>;
// template<class InputIterator, class Allocator>
// map(InputIterator, InputIterator, Allocator)
//   -> map<iter-value-type<InputIterator>, less<iter-value-type<InputIterator>>, Allocator>;
// template<class Key, class Allocator>
// map(initializer_list<Key>, Allocator)
//   -> map<Key, less<Key>, Allocator>;
//
// template<ranges::input_range R, class Compare = less<range-key-type<R>,
//          class Allocator = allocator<range-to-alloc-type<R>>>
//   map(from_range_t, R&&, Compare = Compare(), Allocator = Allocator())
//     -> map<range-key-type<R>, range-mapped-type<R>, Compare, Allocator>; // C++23
//
// template<ranges::input_range R, class Allocator>
//   map(from_range_t, R&&, Allocator)
//     -> map<range-key-type<R>, range-mapped-type<R>, less<range-key-type<R>>, Allocator>; // C++23

#include <algorithm> // std::equal
#include <array>
#include <cassert>
#include <climits> // INT_MAX
#include <functional>
#include <map>
#include <type_traits>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

using P = std::pair<int, long>;
using PC = std::pair<const int, long>;

int main(int, char**)
{
    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr), std::greater<int>());

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::greater<int>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr), std::greater<int>(), test_allocator<PC>(0, 42));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::greater<int>, test_allocator<PC>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 42);
    }

    {
    std::map<int, long> source;
    std::map m(source);
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    std::map<int, long> source;
    std::map m{source};  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    std::map<int, long> source;
    std::map m(source, std::map<int, long>::allocator_type());
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    std::map m{ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} };

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    std::map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::greater<int>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    std::map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, std::greater<int>(), test_allocator<PC>(0, 43));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::greater<int>, test_allocator<PC>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 43);
    }

    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr), test_allocator<PC>(0, 44));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::less<int>, test_allocator<PC>>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 44);
    }

    {
    std::map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, test_allocator<PC>(0, 45));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::less<int>, test_allocator<PC>>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 45);
    }

    {
    // Examples from LWG3025
    std::map m{std::pair{1, 1}, {2, 2}, {3, 3}};
    ASSERT_SAME_TYPE(decltype(m), std::map<int, int>);

    std::map m2{m.begin(), m.end()};
    ASSERT_SAME_TYPE(decltype(m2), std::map<int, int>);
    }

    {
    // Examples from LWG3531
    std::map m1{{std::pair{1, 2}, {3, 4}}, std::less<int>()};
    ASSERT_SAME_TYPE(decltype(m1), std::map<int, int>);

    using value_type = std::pair<const int, int>;
    std::map m2{{value_type{1, 2}, {3, 4}}, std::less<int>()};
    ASSERT_SAME_TYPE(decltype(m2), std::map<int, int>);
    }

#if TEST_STD_VER >= 23
    {
      using Range = std::array<P, 0>;
      using Comp = std::greater<int>;
      using DefaultComp = std::less<int>;
      using Alloc = test_allocator<PC>;

      { // (from_range, range)
        std::map c(std::from_range, Range());
        static_assert(std::is_same_v<decltype(c), std::map<int, long>>);
      }

      { // (from_range, range, comp)
        std::map c(std::from_range, Range(), Comp());
        static_assert(std::is_same_v<decltype(c), std::map<int, long, Comp>>);
      }

      { // (from_range, range, comp, alloc)
        std::map c(std::from_range, Range(), Comp(), Alloc());
        static_assert(std::is_same_v<decltype(c), std::map<int, long, Comp, Alloc>>);
      }

      { // (from_range, range, alloc)
        std::map c(std::from_range, Range(), Alloc());
        static_assert(std::is_same_v<decltype(c), std::map<int, long, DefaultComp, Alloc>>);
      }
    }
#endif

    AssociativeContainerDeductionGuidesSfinaeAway<std::map, std::map<int, long>>();

    return 0;
}
