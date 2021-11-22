//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <unordered_set>

// template<class InputIterator,
//        class Hash = hash<iter-value-type<InputIterator>>,
//        class Pred = equal_to<iter-value-type<InputIterator>>,
//        class Allocator = allocator<iter-value-type<InputIterator>>>
// unordered_multiset(InputIterator, InputIterator, typename see below::size_type = see below,
//                    Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_multiset<iter-value-type<InputIterator>,
//                         Hash, Pred, Allocator>;
//
// template<class T, class Hash = hash<T>,
//        class Pred = equal_to<T>, class Allocator = allocator<T>>
// unordered_multiset(initializer_list<T>, typename see below::size_type = see below,
//                    Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_multiset<T, Hash, Pred, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_multiset(InputIterator, InputIterator, typename see below::size_type, Allocator)
//   -> unordered_multiset<iter-value-type<InputIterator>,
//                         hash<iter-value-type<InputIterator>>,
//                         equal_to<iter-value-type<InputIterator>>,
//                         Allocator>;
//
// template<class InputIterator, class Hash, class Allocator>
// unordered_multiset(InputIterator, InputIterator, typename see below::size_type,
//                    Hash, Allocator)
//   -> unordered_multiset<iter-value-type<InputIterator>, Hash,
//                         equal_to<iter-value-type<InputIterator>>,
//                         Allocator>;
//
// template<class T, class Allocator>
// unordered_multiset(initializer_list<T>, typename see below::size_type, Allocator)
//   -> unordered_multiset<T, hash<T>, equal_to<T>, Allocator>;
//
// template<class T, class Hash, class Allocator>
// unordered_multiset(initializer_list<T>, typename see below::size_type, Hash, Allocator)
//   -> unordered_multiset<T, Hash, equal_to<T>, Allocator>;

#include <algorithm> // is_permutation
#include <cassert>
#include <climits> // INT_MAX
#include <type_traits>
#include <unordered_set>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

int main(int, char**)
{
    const int expected_s[] = {1, 1, 2, 3, INT_MAX};

    {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::unordered_multiset s(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::unordered_multiset s(std::begin(arr), std::end(arr), 42);

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::unordered_multiset s(std::begin(arr), std::end(arr), 42, std::hash<long long>());

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::unordered_multiset s(std::begin(arr), std::end(arr), 42, std::hash<long long>(), test_allocator<int>(0, 40));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>, std::equal_to<int>, test_allocator<int>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    assert(s.get_allocator().get_id() == 40);
    }

    {
    std::unordered_multiset<int, std::hash<long long>, std::equal_to<>, test_allocator<int>> source;
    std::unordered_multiset s(source);
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s.size() == 0);
    }

    {
    std::unordered_multiset<int, std::hash<long long>, std::equal_to<>, test_allocator<int>> source;
    std::unordered_multiset s{source};  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s.size() == 0);
    }

    {
    std::unordered_multiset<int, std::hash<long long>, std::equal_to<>, test_allocator<int>> source;
    std::unordered_multiset s(source, test_allocator<int>(0, 41));
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s.size() == 0);
    assert(s.get_allocator().get_id() == 41);
    }

    {
    std::unordered_multiset<int, std::hash<long long>, std::equal_to<>, test_allocator<int>> source;
    std::unordered_multiset s{source, test_allocator<int>(0, 42)};  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s.size() == 0);
    assert(s.get_allocator().get_id() == 42);
    }

    {
    std::unordered_multiset s{ 1, 2, 1, INT_MAX, 3 };

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    std::unordered_multiset s({ 1, 2, 1, INT_MAX, 3 }, 42);

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    std::unordered_multiset s({ 1, 2, 1, INT_MAX, 3 }, 42, std::hash<long long>());

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    std::unordered_multiset s({ 1, 2, 1, INT_MAX, 3 }, 42, std::hash<long long>(), std::equal_to<>());

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>, std::equal_to<>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    }

    {
    std::unordered_multiset s({ 1, 2, 1, INT_MAX, 3 }, 42, std::hash<long long>(), std::equal_to<>(), test_allocator<int>(0, 43));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>, std::equal_to<>, test_allocator<int>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    assert(s.get_allocator().get_id() == 43);
    }

    {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::unordered_multiset s(std::begin(arr), std::end(arr), 42, test_allocator<int>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, test_allocator<int>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    assert(s.get_allocator().get_id() == 44);
    }

    {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::unordered_multiset s(std::begin(arr), std::end(arr), 42, std::hash<long long>(), test_allocator<int>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>, std::equal_to<int>, test_allocator<int>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    assert(s.get_allocator().get_id() == 44);
    }

    {
    std::unordered_multiset s({ 1, 2, 1, INT_MAX, 3 }, 42, test_allocator<int>(0, 43));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, test_allocator<int>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    assert(s.get_allocator().get_id() == 43);
    }

    {
    std::unordered_multiset s({ 1, 2, 1, INT_MAX, 3 }, 42, std::hash<long long>(), test_allocator<int>(0, 42));

    ASSERT_SAME_TYPE(decltype(s), std::unordered_multiset<int, std::hash<long long>, std::equal_to<int>, test_allocator<int>>);
    assert(std::is_permutation(s.begin(), s.end(), std::begin(expected_s), std::end(expected_s)));
    assert(s.get_allocator().get_id() == 42);
    }

    UnorderedContainerDeductionGuidesSfinaeAway<std::unordered_multiset, std::unordered_multiset<int>>();

    return 0;
}
