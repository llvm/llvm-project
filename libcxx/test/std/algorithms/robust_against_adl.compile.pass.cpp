//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

#include <algorithm>
#include <cstddef>
#include <functional>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };

template<class>
struct ConvertibleToIntegral {
    TEST_CONSTEXPR operator int() const { return 1; }
};

struct Tester {
    using Element = Holder<Incomplete>*;
    Element data[10] = {};
};

struct UnaryVoid { TEST_CONSTEXPR_CXX14 void operator()(void*) const {} };
struct UnaryTrue { TEST_CONSTEXPR bool operator()(void*) const { return true; } };
struct NullaryValue { TEST_CONSTEXPR std::nullptr_t operator()() const { return nullptr; } };
struct UnaryTransform { TEST_CONSTEXPR std::nullptr_t operator()(void*) const { return nullptr; } };
struct BinaryTransform { TEST_CONSTEXPR std::nullptr_t operator()(void*, void*) const { return nullptr; } };

TEST_CONSTEXPR_CXX20 bool all_the_algorithms()
{
    Tester t;
    Tester u;
    Holder<Incomplete> **first = t.data;
    Holder<Incomplete> **mid = t.data+5;
    Holder<Incomplete> **last = t.data+10;
    Holder<Incomplete> **first2 = u.data;
    Holder<Incomplete> **mid2 = u.data+5;
    Holder<Incomplete> **last2 = u.data+10;
    Tester::Element value = nullptr;
    ConvertibleToIntegral<Tester::Element> count;

    (void)std::adjacent_find(first, last);
    (void)std::adjacent_find(first, last, std::equal_to<void*>());
#if TEST_STD_VER >= 11
    (void)std::all_of(first, last, UnaryTrue());
    (void)std::any_of(first, last, UnaryTrue());
#endif
    (void)std::binary_search(first, last, value);
    (void)std::binary_search(first, last, value, std::less<void*>());
#if TEST_STD_VER > 17
    (void)std::clamp(value, value, value);
    (void)std::clamp(value, value, value, std::less<void*>());
#endif
    (void)std::copy(first, last, first2);
    (void)std::copy_backward(first, last, last2);
    // TODO FIXME (void)std::copy_n(first, count, first2);
    (void)std::count(first, last, value);
    (void)std::count_if(first, last, UnaryTrue());
    (void)std::distance(first, last);
    (void)std::equal(first, last, first2);
    (void)std::equal(first, last, first2, std::equal_to<void*>());
#if TEST_STD_VER > 11
    (void)std::equal(first, last, first2, last2);
    (void)std::equal(first, last, first2, last2, std::equal_to<void*>());
#endif
    (void)std::equal_range(first, last, value);
    (void)std::equal_range(first, last, value, std::less<void*>());
    (void)std::fill(first, last, value);
    (void)std::fill_n(first, count, value);
    (void)std::find(first, last, value);
    // TODO FIXME (void)std::find_end(first, last, first2, mid2);
    // TODO FIXME (void)std::find_end(first, last, first2, mid2, std::equal_to<void*>());
    (void)std::find_if(first, last, UnaryTrue());
    (void)std::find_if_not(first, last, UnaryTrue());
    (void)std::for_each(first, last, UnaryVoid());
#if TEST_STD_VER > 14
    (void)std::for_each_n(first, count, UnaryVoid());
#endif
    (void)std::generate(first, last, NullaryValue());
    (void)std::generate_n(first, count, NullaryValue());
    (void)std::includes(first, last, first2, last2);
    (void)std::includes(first, last, first2, last2, std::less<void*>());
    (void)std::is_heap(first, last);
    (void)std::is_heap(first, last, std::less<void*>());
    (void)std::is_heap_until(first, last);
    (void)std::is_heap_until(first, last, std::less<void*>());
    (void)std::is_partitioned(first, last, UnaryTrue());
    (void)std::is_permutation(first, last, first2);
    (void)std::is_permutation(first, last, first2, std::equal_to<void*>());
#if TEST_STD_VER > 11
    (void)std::is_permutation(first, last, first2, last2);
    (void)std::is_permutation(first, last, first2, last2, std::equal_to<void*>());
#endif
    (void)std::is_sorted(first, last);
    (void)std::is_sorted(first, last, std::less<void*>());
    (void)std::is_sorted_until(first, last);
    (void)std::is_sorted_until(first, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::inplace_merge(first, mid, last);
    // RELIES ON ADL SWAP (void)std::inplace_merge(first, mid, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::iter_swap(first, mid);
    (void)std::lexicographical_compare(first, last, first2, last2);
    (void)std::lexicographical_compare(first, last, first2, last2, std::less<void*>());
    // TODO: lexicographical_compare_three_way
    (void)std::lower_bound(first, last, value);
    (void)std::lower_bound(first, last, value, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::make_heap(first, last);
    // RELIES ON ADL SWAP (void)std::make_heap(first, last, std::less<void*>());
    (void)std::max(value, value);
    (void)std::max(value, value, std::less<void*>());
#if TEST_STD_VER >= 11
    (void)std::max({ value, value });
    (void)std::max({ value, value }, std::less<void*>());
#endif
    (void)std::max_element(first, last);
    (void)std::max_element(first, last, std::less<void*>());
    (void)std::merge(first, mid, mid, last, first2);
    (void)std::merge(first, mid, mid, last, first2, std::less<void*>());
    (void)std::min(value, value);
    (void)std::min(value, value, std::less<void*>());
#if TEST_STD_VER >= 11
    (void)std::min({ value, value });
    (void)std::min({ value, value }, std::less<void*>());
#endif
    (void)std::min_element(first, last);
    (void)std::min_element(first, last, std::less<void*>());
    (void)std::minmax(value, value);
    (void)std::minmax(value, value, std::less<void*>());
#if TEST_STD_VER >= 11
    (void)std::minmax({ value, value });
    (void)std::minmax({ value, value }, std::less<void*>());
#endif
    (void)std::minmax_element(first, last);
    (void)std::minmax_element(first, last, std::less<void*>());
    (void)std::mismatch(first, last, first2);
    (void)std::mismatch(first, last, first2, std::equal_to<void*>());
#if TEST_STD_VER > 11
    (void)std::mismatch(first, last, first2, last2);
    (void)std::mismatch(first, last, first2, last2, std::equal_to<void*>());
#endif
    (void)std::move(first, last, first2);
    (void)std::move_backward(first, last, last2);
    // RELIES ON ADL SWAP (void)std::next_permutation(first, last);
    // RELIES ON ADL SWAP (void)std::next_permutation(first, last, std::less<void*>());
#if TEST_STD_VER >= 11
    (void)std::none_of(first, last, UnaryTrue());
#endif
    // RELIES ON ADL SWAP (void)std::nth_element(first, mid, last);
    // RELIES ON ADL SWAP (void)std::nth_element(first, mid, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::partial_sort(first, mid, last);
    // RELIES ON ADL SWAP (void)std::partial_sort(first, mid, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::partial_sort_copy(first, last, first2, mid2);
    // RELIES ON ADL SWAP (void)std::partial_sort_copy(first, last, first2, mid2, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::partition(first, last, UnaryTrue());
    (void)std::partition_copy(first, last, first2, last2, UnaryTrue());
    (void)std::partition_point(first, last, UnaryTrue());
    // RELIES ON ADL SWAP (void)std::pop_heap(first, last);
    // RELIES ON ADL SWAP (void)std::pop_heap(first, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::prev_permutation(first, last);
    // RELIES ON ADL SWAP (void)std::prev_permutation(first, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::push_heap(first, last);
    // RELIES ON ADL SWAP (void)std::push_heap(first, last, std::less<void*>());
    (void)std::remove(first, last, value);
    (void)std::remove_copy(first, last, first2, value);
    (void)std::remove_copy_if(first, last, first2, UnaryTrue());
    (void)std::remove_if(first, last, UnaryTrue());
    (void)std::replace(first, last, value, value);
    (void)std::replace_copy(first, last, first2, value, value);
    (void)std::replace_copy_if(first, last, first2, UnaryTrue(), value);
    (void)std::replace_if(first, last, UnaryTrue(), value);
    // RELIES ON ADL SWAP (void)std::reverse(first, last);
    // RELIES ON ADL SWAP (void)std::reverse_copy(first, last, first2);
    // RELIES ON ADL SWAP (void)std::rotate(first, mid, last);
    (void)std::rotate_copy(first, mid, last, first2);
    (void)std::search(first, last, first2, mid2);
    (void)std::search(first, last, first2, mid2, std::equal_to<void*>());
    (void)std::search_n(first, last, count, value);
    (void)std::search_n(first, last, count, value, std::equal_to<void*>());
    (void)std::set_difference(first, mid, mid, last, first2);
    (void)std::set_difference(first, mid, mid, last, first2, std::less<void*>());
    (void)std::set_intersection(first, mid, mid, last, first2);
    (void)std::set_intersection(first, mid, mid, last, first2, std::less<void*>());
    (void)std::set_symmetric_difference(first, mid, mid, last, first2);
    (void)std::set_symmetric_difference(first, mid, mid, last, first2, std::less<void*>());
    (void)std::set_union(first, mid, mid, last, first2);
    (void)std::set_union(first, mid, mid, last, first2, std::less<void*>());
#if TEST_STD_VER > 17
    (void)std::shift_left(first, last, count);
    (void)std::shift_right(first, last, count);
#endif
    // RELIES ON ADL SWAP (void)std::sort(first, last);
    // RELIES ON ADL SWAP (void)std::sort(first, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::sort_heap(first, last);
    // RELIES ON ADL SWAP (void)std::sort_heap(first, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::stable_partition(first, last, UnaryTrue());
    // RELIES ON ADL SWAP (void)std::stable_sort(first, last);
    // RELIES ON ADL SWAP (void)std::stable_sort(first, last, std::less<void*>());
    // RELIES ON ADL SWAP (void)std::swap_ranges(first, last, first2);
    (void)std::transform(first, last, first2, UnaryTransform());
    (void)std::transform(first, mid, mid, first2, BinaryTransform());
    (void)std::unique(first, last);
    (void)std::unique(first, last, std::equal_to<void*>());
    (void)std::unique_copy(first, last, first2);
    (void)std::unique_copy(first, last, first2, std::equal_to<void*>());
    (void)std::upper_bound(first, last, value);
    (void)std::upper_bound(first, last, value, std::less<void*>());

    return true;
}

void test()
{
    all_the_algorithms();
#if TEST_STD_VER > 17
    static_assert(all_the_algorithms());
#endif
}
