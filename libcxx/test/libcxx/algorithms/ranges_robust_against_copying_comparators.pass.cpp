//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// this test checks that the comparators in the ranges algorithms aren't copied/moved

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <type_traits>

#include "test_macros.h"

struct Less {
    int *copies_;
    constexpr explicit Less(int *copies) : copies_(copies) {}
    constexpr Less(const Less& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr Less& operator=(const Less&) = default;
    constexpr bool operator()(void*, void*) const { return false; }
};

struct Equal {
    int *copies_;
    constexpr explicit Equal(int *copies) : copies_(copies) {}
    constexpr Equal(const Equal& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr Equal& operator=(const Equal&) = default;
    constexpr bool operator()(void*, void*) const { return true; }
};

struct UnaryVoid {
    int *copies_;
    constexpr explicit UnaryVoid(int *copies) : copies_(copies) {}
    constexpr UnaryVoid(const UnaryVoid& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr UnaryVoid& operator=(const UnaryVoid&) = default;
    constexpr void operator()(void*) const {}
};

struct UnaryTrue {
    int *copies_;
    constexpr explicit UnaryTrue(int *copies) : copies_(copies) {}
    constexpr UnaryTrue(const UnaryTrue& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr UnaryTrue& operator=(const UnaryTrue&) = default;
    constexpr bool operator()(void*) const { return true; }
};

struct NullaryValue {
    int *copies_;
    constexpr explicit NullaryValue(int *copies) : copies_(copies) {}
    constexpr NullaryValue(const NullaryValue& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr NullaryValue& operator=(const NullaryValue&) = default;
    constexpr std::nullptr_t operator()() const { return nullptr; }
};

struct UnaryTransform {
    int *copies_;
    constexpr explicit UnaryTransform(int *copies) : copies_(copies) {}
    constexpr UnaryTransform(const UnaryTransform& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr UnaryTransform& operator=(const UnaryTransform&) = default;
    constexpr std::nullptr_t operator()(void*) const { return nullptr; }
};

struct BinaryTransform {
    int *copies_;
    constexpr explicit BinaryTransform(int *copies) : copies_(copies) {}
    constexpr BinaryTransform(const BinaryTransform& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr BinaryTransform& operator=(const BinaryTransform&) = default;
    constexpr std::nullptr_t operator()(void*, void*) const { return nullptr; }
};

constexpr bool all_the_algorithms()
{
    void *a[10] = {};
    void *b[10] = {};
    void *half[5] = {};
    void **first = a;
    void **mid = a+5;
    void **last = a+10;
    void **first2 = b;
    void **mid2 = b+5;
    void **last2 = b+10;
    void *value = nullptr;
    int count = 1;

    int copies = 0;
    (void)std::ranges::adjacent_find(first, last, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::adjacent_find(a, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::all_of(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::all_of(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::any_of(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::any_of(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::binary_search(first, last, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::binary_search(a, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::clamp(value, value, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::count_if(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::count_if(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::copy_if(first, last, first2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::copy_if(a, first2, UnaryTrue(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 23
    (void)std::ranges::ends_with(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::ends_with(a, b, Equal(&copies)); assert(copies == 0);
#endif
    (void)std::ranges::equal(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::equal(a, b, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::equal_range(first, last, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::equal_range(a, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::find_end(first, last, first2, mid2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::find_end(a, b, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::find_first_of(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::find_first_of(a, b, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::find_if(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::find_if(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::find_if_not(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::find_if_not(a, UnaryTrue(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 23
    (void)std::ranges::find_last_if(first, last, UnaryTrue(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if(a, UnaryTrue(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if_not(first, last, UnaryTrue(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if_not(a, UnaryTrue(&copies));
    assert(copies == 0);
#endif
    (void)std::ranges::for_each(first, last, UnaryVoid(&copies)); assert(copies == 1); copies = 0;
    (void)std::ranges::for_each(a, UnaryVoid(&copies)); assert(copies == 1); copies = 0;
    (void)std::ranges::for_each_n(first, count, UnaryVoid(&copies)); assert(copies == 1); copies = 0;
    (void)std::ranges::generate(first, last, NullaryValue(&copies)); assert(copies == 0);
    (void)std::ranges::generate(a, NullaryValue(&copies)); assert(copies == 0);
    (void)std::ranges::generate_n(first, count, NullaryValue(&copies)); assert(copies == 0);
    (void)std::ranges::includes(first, last, first2, last2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::includes(a, b, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap_until(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap_until(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_partitioned(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::is_partitioned(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::is_permutation(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::is_permutation(a, b, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted_until(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted_until(a, Less(&copies)); assert(copies == 0);
    if (!std::is_constant_evaluated()) { (void)std::ranges::inplace_merge(first, mid, last, Less(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::inplace_merge(a, mid, Less(&copies)); assert(copies == 0); }
    (void)std::ranges::lexicographical_compare(first, last, first2, last2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::lexicographical_compare(a, b, Less(&copies)); assert(copies == 0);
    (void)std::ranges::lower_bound(first, last, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::lower_bound(a, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::make_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::make_heap(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::max(value, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::max({ value, value }, Less(&copies)); assert(copies == 0);
    (void)std::ranges::max(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::max_element(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::max_element(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::merge(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::merge(half, half, b, Less(&copies)); assert(copies == 0);
    (void)std::ranges::min(value, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::min({ value, value }, Less(&copies)); assert(copies == 0);
    (void)std::ranges::min(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::min_element(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::min_element(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::minmax(value, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::minmax({ value, value }, Less(&copies)); assert(copies == 0);
    (void)std::ranges::minmax(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::minmax_element(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::minmax_element(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::mismatch(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::mismatch(a, b, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::next_permutation(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::next_permutation(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::none_of(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::none_of(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::nth_element(first, mid, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::nth_element(a, mid, Less(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort(first, mid, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort(a, mid, Less(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort_copy(first, last, first2, mid2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort_copy(a, b, Less(&copies)); assert(copies == 0);
    (void)std::ranges::partition(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::partition(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::partition_copy(first, last, first2, last2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::partition_copy(a, first2, last2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::partition_point(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::partition_point(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::pop_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::pop_heap(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::prev_permutation(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::prev_permutation(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::push_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::push_heap(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::remove_copy_if(first, last, first2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::remove_copy_if(a, first2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::remove_if(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::remove_if(a, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::ranges::replace_copy_if(first, last, first2, UnaryTrue(&copies), value); assert(copies == 0);
    (void)std::ranges::replace_copy_if(a, first2, UnaryTrue(&copies), value); assert(copies == 0);
    (void)std::ranges::replace_if(first, last, UnaryTrue(&copies), value); assert(copies == 0);
    (void)std::ranges::replace_if(a, UnaryTrue(&copies), value); assert(copies == 0);
    (void)std::ranges::search(first, last, first2, mid2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::search(a, b, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::search_n(first, last, count, value, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::search_n(a, count, value, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::set_difference(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_difference(a, b, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_intersection(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_intersection(a, b, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_symmetric_difference(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_symmetric_difference(a, b, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_union(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::set_union(a, b, first2, Less(&copies)); assert(copies == 0);
    (void)std::ranges::sort(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::sort(a, Less(&copies)); assert(copies == 0);
    (void)std::ranges::sort_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::ranges::sort_heap(a, Less(&copies)); assert(copies == 0);
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_partition(first, last, UnaryTrue(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_partition(a, UnaryTrue(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_sort(first, last, Less(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_sort(a, Less(&copies)); assert(copies == 0); }
#if TEST_STD_VER > 20
    (void)std::ranges::starts_with(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::starts_with(a, b, Equal(&copies)); assert(copies == 0);
#endif
    (void)std::ranges::transform(first, last, first2, UnaryTransform(&copies)); assert(copies == 0);
    (void)std::ranges::transform(a, first2, UnaryTransform(&copies)); assert(copies == 0);
    (void)std::ranges::transform(first, mid, mid, last, first2, BinaryTransform(&copies)); assert(copies == 0);
    (void)std::ranges::transform(a, b, first2, BinaryTransform(&copies)); assert(copies == 0);
    (void)std::ranges::unique(first, last, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::unique(a, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::unique_copy(first, last, first2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::unique_copy(a, first2, Equal(&copies)); assert(copies == 0);
    (void)std::ranges::upper_bound(first, last, value, Less(&copies)); assert(copies == 0);
    (void)std::ranges::upper_bound(a, value, Less(&copies)); assert(copies == 0);

    return true;
}

int main(int, char**)
{
    all_the_algorithms();
    static_assert(all_the_algorithms());

    return 0;
}
