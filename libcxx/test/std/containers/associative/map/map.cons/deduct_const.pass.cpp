//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

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

#include <algorithm> // std::equal
#include <cassert>
#include <climits> // INT_MAX
#include <functional>
#include <map>
#include <type_traits>

#include "test_allocator.h"

using P = std::pair<int, long>;
using PC = std::pair<const int, long>;
using PCC = std::pair<const int, const long>;

int main(int, char**)
{
    {
    const PCC arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, const long>);
    const PCC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    const PCC arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr), std::greater<int>());

    ASSERT_SAME_TYPE(decltype(m), std::map<int, const long, std::greater<int>>);
    const PCC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1, 1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    const PCC arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    std::map m(std::begin(arr), std::end(arr), std::greater<int>(), test_allocator<PCC>(0, 42));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, const long, std::greater<int>, test_allocator<PCC>>);
    const PCC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1, 1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 42);
    }

    {
    std::map m{ PC{1,1L}, PC{2,2L}, PC{1,1L}, PC{INT_MAX,1L}, PC{3,1L} };

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long>);
    const PC expected_m[] = { {1, 1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    std::map m({ PC{1,1L}, PC{2,2L}, PC{1,1L}, PC{INT_MAX,1L}, PC{3,1L} }, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::greater<int>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1, 1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    }

    {
    std::map m({ PC{1,1L}, PC{2,2L}, PC{1,1L}, PC{INT_MAX,1L}, PC{3,1L} }, std::greater<int>(), test_allocator<PC>(0, 43));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::greater<int>, test_allocator<PC>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1, 1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 43);
    }

    {
    std::map m({ PC{1,1L}, PC{2,2L}, PC{1,1L}, PC{INT_MAX,1L}, PC{3,1L} }, test_allocator<PC>(0, 45));

    ASSERT_SAME_TYPE(decltype(m), std::map<int, long, std::less<int>, test_allocator<PC>>);
    const PC expected_m[] = { {1, 1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(std::equal(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 45);
    }

    return 0;
}
