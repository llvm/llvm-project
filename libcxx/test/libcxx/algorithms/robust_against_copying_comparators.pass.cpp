//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

struct Less {
    int *copies_;
    TEST_CONSTEXPR explicit Less(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 Less(const Less& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 Less& operator=(const Less&) = default;
    TEST_CONSTEXPR bool operator()(void*, void*) const { return false; }
};

struct Equal {
    int *copies_;
    TEST_CONSTEXPR explicit Equal(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 Equal(const Equal& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 Equal& operator=(const Equal&) = default;
    TEST_CONSTEXPR bool operator()(void*, void*) const { return true; }
};

struct UnaryVoid {
    int *copies_;
    TEST_CONSTEXPR explicit UnaryVoid(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 UnaryVoid(const UnaryVoid& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 UnaryVoid& operator=(const UnaryVoid&) = default;
    TEST_CONSTEXPR_CXX14 void operator()(void*) const {}
};

struct UnaryTrue {
    int *copies_;
    TEST_CONSTEXPR explicit UnaryTrue(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 UnaryTrue(const UnaryTrue& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 UnaryTrue& operator=(const UnaryTrue&) = default;
    TEST_CONSTEXPR bool operator()(void*) const { return true; }
};

struct NullaryValue {
    int *copies_;
    TEST_CONSTEXPR explicit NullaryValue(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 NullaryValue(const NullaryValue& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 NullaryValue& operator=(const NullaryValue&) = default;
    TEST_CONSTEXPR std::nullptr_t operator()() const { return nullptr; }
};

struct UnaryTransform {
    int *copies_;
    TEST_CONSTEXPR explicit UnaryTransform(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 UnaryTransform(const UnaryTransform& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 UnaryTransform& operator=(const UnaryTransform&) = default;
    TEST_CONSTEXPR std::nullptr_t operator()(void*) const { return nullptr; }
};

struct BinaryTransform {
    int *copies_;
    TEST_CONSTEXPR explicit BinaryTransform(int *copies) : copies_(copies) {}
    TEST_CONSTEXPR_CXX14 BinaryTransform(const BinaryTransform& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    TEST_CONSTEXPR_CXX14 BinaryTransform& operator=(const BinaryTransform&) = default;
    TEST_CONSTEXPR std::nullptr_t operator()(void*, void*) const { return nullptr; }
};

TEST_CONSTEXPR_CXX20 bool all_the_algorithms()
{
    void *a[10] = {};
    void *b[10] = {};
    void **first = a;
    void **mid = a+5;
    void **last = a+10;
    void **first2 = b;
    void **mid2 = b+5;
    void **last2 = b+10;
    void *value = nullptr;
    int count = 1;

    int copies = 0;
    (void)std::adjacent_find(first, last, Equal(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 11
    (void)std::all_of(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::any_of(first, last, UnaryTrue(&copies)); assert(copies == 0);
#endif
    (void)std::binary_search(first, last, value, Less(&copies)); assert(copies == 0);
#if TEST_STD_VER > 17
    (void)std::clamp(value, value, value, Less(&copies)); assert(copies == 0);
#endif
    (void)std::count_if(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::equal(first, last, first2, Equal(&copies)); assert(copies == 0);
#if TEST_STD_VER > 11
    (void)std::equal(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
#endif
    (void)std::equal_range(first, last, value, Less(&copies)); assert(copies == 0);
    (void)std::find_end(first, last, first2, mid2, Equal(&copies)); assert(copies == 0);
    (void)std::find_if(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::find_if_not(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::for_each(first, last, UnaryVoid(&copies)); assert(copies == 1); copies = 0;
#if TEST_STD_VER > 14
    (void)std::for_each_n(first, count, UnaryVoid(&copies)); assert(copies == 0);
#endif
    (void)std::generate(first, last, NullaryValue(&copies)); assert(copies == 0);
    (void)std::generate_n(first, count, NullaryValue(&copies)); assert(copies == 0);
    (void)std::includes(first, last, first2, last2, Less(&copies)); assert(copies == 0);
    (void)std::is_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::is_heap_until(first, last, Less(&copies)); assert(copies == 0);
    (void)std::is_partitioned(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::is_permutation(first, last, first2, Equal(&copies)); assert(copies == 0);
#if TEST_STD_VER > 11
    (void)std::is_permutation(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
#endif
    (void)std::is_sorted(first, last, Less(&copies)); assert(copies == 0);
    (void)std::is_sorted_until(first, last, Less(&copies)); assert(copies == 0);
    if (!TEST_IS_CONSTANT_EVALUATED) { (void)std::inplace_merge(first, mid, last, Less(&copies)); assert(copies == 0); }
    (void)std::lexicographical_compare(first, last, first2, last2, Less(&copies)); assert(copies == 0);
    // TODO: lexicographical_compare_three_way
    (void)std::lower_bound(first, last, value, Less(&copies)); assert(copies == 0);
    (void)std::make_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::max(value, value, Less(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 11
    (void)std::max({ value, value }, Less(&copies)); assert(copies == 0);
#endif
    (void)std::max_element(first, last, Less(&copies)); assert(copies == 0);
    (void)std::merge(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::min(value, value, Less(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 11
    (void)std::min({ value, value }, Less(&copies)); assert(copies == 0);
#endif
    (void)std::min_element(first, last, Less(&copies)); assert(copies == 0);
    (void)std::minmax(value, value, Less(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 11
    (void)std::minmax({ value, value }, Less(&copies)); assert(copies == 0);
#endif
    (void)std::minmax_element(first, last, Less(&copies)); assert(copies == 0);
    (void)std::mismatch(first, last, first2, Equal(&copies)); assert(copies == 0);
#if TEST_STD_VER > 11
    (void)std::mismatch(first, last, first2, last2, Equal(&copies)); assert(copies == 0);
#endif
    (void)std::next_permutation(first, last, Less(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 11
    (void)std::none_of(first, last, UnaryTrue(&copies)); assert(copies == 0);
#endif
    (void)std::nth_element(first, mid, last, Less(&copies)); assert(copies == 0);
    (void)std::partial_sort(first, mid, last, Less(&copies)); assert(copies == 0);
    (void)std::partial_sort_copy(first, last, first2, mid2, Less(&copies)); assert(copies == 0);
    (void)std::partition(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::partition_copy(first, last, first2, last2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::partition_point(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::pop_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::prev_permutation(first, last, Less(&copies)); assert(copies == 0);
    (void)std::push_heap(first, last, Less(&copies)); assert(copies == 0);
    (void)std::remove_copy_if(first, last, first2, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::remove_if(first, last, UnaryTrue(&copies)); assert(copies == 0);
    (void)std::replace_copy_if(first, last, first2, UnaryTrue(&copies), value); assert(copies == 0);
    (void)std::replace_if(first, last, UnaryTrue(&copies), value); assert(copies == 0);
    (void)std::search(first, last, first2, mid2, Equal(&copies)); assert(copies == 0);
    (void)std::search_n(first, last, count, value, Equal(&copies)); assert(copies == 0);
    (void)std::set_difference(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::set_intersection(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::set_symmetric_difference(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::set_union(first, mid, mid, last, first2, Less(&copies)); assert(copies == 0);
    (void)std::sort(first, last, Less(&copies)); assert(copies == 0);
    (void)std::sort_heap(first, last, Less(&copies)); assert(copies == 0);
    if (!TEST_IS_CONSTANT_EVALUATED) { (void)std::stable_partition(first, last, UnaryTrue(&copies)); assert(copies == 0); }
    if (!TEST_IS_CONSTANT_EVALUATED) { (void)std::stable_sort(first, last, Less(&copies)); assert(copies == 0); }
    (void)std::transform(first, last, first2, UnaryTransform(&copies)); assert(copies == 0);
    (void)std::transform(first, mid, mid, first2, BinaryTransform(&copies)); assert(copies == 0);
    (void)std::unique(first, last, Equal(&copies)); assert(copies == 0);
    (void)std::unique_copy(first, last, first2, Equal(&copies)); assert(copies == 0);
    (void)std::upper_bound(first, last, value, Less(&copies)); assert(copies == 0);

    return true;
}

int main(int, char**)
{
    all_the_algorithms();
#if TEST_STD_VER > 17
    static_assert(all_the_algorithms());
#endif

    return 0;
}
