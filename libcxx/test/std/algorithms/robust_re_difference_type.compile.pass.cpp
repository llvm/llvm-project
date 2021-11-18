//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>
//   In the description of the algorithms,
//   given an iterator a whose difference type is D,
//   and an expression n of integer-like type other than cv D,
//   the semantics of a + n and a - n are, respectively,
//   those of a + D(n) and a - D(n).

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>

#include "test_macros.h"

// This iterator rejects expressions like (a + n) and (a - n)
// whenever n is of any type other than difference_type.
//
template<class It, class DifferenceType>
class PickyIterator {
    It it_;
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::iterator_traits<It>::value_type;
    using difference_type = DifferenceType;
    using pointer = It;
    using reference = typename std::iterator_traits<It>::reference;

    TEST_CONSTEXPR_CXX14 PickyIterator() = default;
    TEST_CONSTEXPR_CXX14 explicit PickyIterator(It it) : it_(it) {}
    TEST_CONSTEXPR_CXX14 reference operator*() const {return *it_;}
    TEST_CONSTEXPR_CXX14 pointer operator->() const {return it_;}
    TEST_CONSTEXPR_CXX14 reference operator[](difference_type n) const {return it_[n];}

    friend TEST_CONSTEXPR_CXX14 bool operator==(PickyIterator a, PickyIterator b) { return a.it_ == b.it_; }
    friend TEST_CONSTEXPR_CXX14 bool operator!=(PickyIterator a, PickyIterator b) { return a.it_ != b.it_; }
    friend TEST_CONSTEXPR_CXX14 bool operator<(PickyIterator a, PickyIterator b) { return a.it_ < b.it_; }
    friend TEST_CONSTEXPR_CXX14 bool operator>(PickyIterator a, PickyIterator b) { return a.it_ > b.it_; }
    friend TEST_CONSTEXPR_CXX14 bool operator<=(PickyIterator a, PickyIterator b) { return a.it_ <= b.it_; }
    friend TEST_CONSTEXPR_CXX14 bool operator>=(PickyIterator a, PickyIterator b) { return a.it_ >= b.it_; }

    TEST_CONSTEXPR_CXX14 PickyIterator& operator++() {++it_; return *this;}
    TEST_CONSTEXPR_CXX14 PickyIterator operator++(int) {auto tmp = *this; ++(*this); return tmp;}
    TEST_CONSTEXPR_CXX14 PickyIterator& operator--() {--it_; return *this;}
    TEST_CONSTEXPR_CXX14 PickyIterator operator--(int) {auto tmp = *this; --(*this); return tmp;}

    TEST_CONSTEXPR_CXX14 PickyIterator& operator+=(difference_type n) {it_ += n; return *this;}
    TEST_CONSTEXPR_CXX14 PickyIterator& operator-=(difference_type n) {it_ -= n; return *this;}
    friend TEST_CONSTEXPR_CXX14 PickyIterator operator+(PickyIterator it, difference_type n) {it += n; return it;}
    friend TEST_CONSTEXPR_CXX14 PickyIterator operator+(difference_type n, PickyIterator it) {it += n; return it;}
    friend TEST_CONSTEXPR_CXX14 PickyIterator operator-(PickyIterator it, difference_type n) {it -= n; return it;}
    friend TEST_CONSTEXPR_CXX14 difference_type operator-(PickyIterator it, PickyIterator jt) {return it.it_ - jt.it_;}

    template<class X> void operator+=(X) = delete;
    template<class X> void operator-=(X) = delete;
    template<class X> friend void operator+(PickyIterator, X) = delete;
    template<class X> friend void operator+(X, PickyIterator) = delete;
    template<class X> friend void operator-(PickyIterator, X) = delete;
};

struct UnaryVoid { TEST_CONSTEXPR_CXX14 void operator()(void*) const {} };
struct UnaryTrue { TEST_CONSTEXPR bool operator()(void*) const { return true; } };
struct NullaryValue { TEST_CONSTEXPR std::nullptr_t operator()() const { return nullptr; } };
struct UnaryTransform { TEST_CONSTEXPR std::nullptr_t operator()(void*) const { return nullptr; } };
struct BinaryTransform { TEST_CONSTEXPR std::nullptr_t operator()(void*, void*) const { return nullptr; } };

TEST_CONSTEXPR_CXX20 bool all_the_algorithms()
{
    void *a[10] = {};
    void *b[10] = {};
    auto first = PickyIterator<void**, long>(a);
    auto mid = PickyIterator<void**, long>(a+5);
    auto last = PickyIterator<void**, long>(a+10);
    auto first2 = PickyIterator<void**, long long>(b);
    auto mid2 = PickyIterator<void**, long long>(b+5);
    auto last2 = PickyIterator<void**, long long>(b+10);
    void *value = nullptr;
    int count = 1;

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
    (void)std::copy_n(first, count, first2);
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
    (void)std::find_end(first, last, first2, mid2);
    (void)std::find_end(first, last, first2, mid2, std::equal_to<void*>());
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
    if (!TEST_IS_CONSTANT_EVALUATED) (void)std::inplace_merge(first, mid, last);
    if (!TEST_IS_CONSTANT_EVALUATED) (void)std::inplace_merge(first, mid, last, std::less<void*>());
    (void)std::iter_swap(first, mid);
    (void)std::lexicographical_compare(first, last, first2, last2);
    (void)std::lexicographical_compare(first, last, first2, last2, std::less<void*>());
    // TODO: lexicographical_compare_three_way
    (void)std::lower_bound(first, last, value);
    (void)std::lower_bound(first, last, value, std::less<void*>());
    (void)std::make_heap(first, last);
    (void)std::make_heap(first, last, std::less<void*>());
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
    (void)std::next_permutation(first, last);
    (void)std::next_permutation(first, last, std::less<void*>());
    (void)std::none_of(first, last, UnaryTrue());
    (void)std::nth_element(first, mid, last);
    (void)std::nth_element(first, mid, last, std::less<void*>());
    (void)std::partial_sort(first, mid, last);
    (void)std::partial_sort(first, mid, last, std::less<void*>());
    (void)std::partial_sort_copy(first, last, first2, mid2);
    (void)std::partial_sort_copy(first, last, first2, mid2, std::less<void*>());
    (void)std::partition(first, last, UnaryTrue());
    (void)std::partition_copy(first, last, first2, last2, UnaryTrue());
    (void)std::partition_point(first, last, UnaryTrue());
    (void)std::pop_heap(first, last);
    (void)std::pop_heap(first, last, std::less<void*>());
    (void)std::prev_permutation(first, last);
    (void)std::prev_permutation(first, last, std::less<void*>());
    (void)std::push_heap(first, last);
    (void)std::push_heap(first, last, std::less<void*>());
    (void)std::remove(first, last, value);
    (void)std::remove_copy(first, last, first2, value);
    (void)std::remove_copy_if(first, last, first2, UnaryTrue());
    (void)std::remove_if(first, last, UnaryTrue());
    (void)std::replace(first, last, value, value);
    (void)std::replace_copy(first, last, first2, value, value);
    (void)std::replace_copy_if(first, last, first2, UnaryTrue(), value);
    (void)std::replace_if(first, last, UnaryTrue(), value);
    (void)std::reverse(first, last);
    (void)std::reverse_copy(first, last, first2);
    (void)std::rotate(first, mid, last);
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
    (void)std::sort(first, last);
    (void)std::sort(first, last, std::less<void*>());
    (void)std::sort_heap(first, last);
    (void)std::sort_heap(first, last, std::less<void*>());
    if (!TEST_IS_CONSTANT_EVALUATED) (void)std::stable_partition(first, last, UnaryTrue());
    if (!TEST_IS_CONSTANT_EVALUATED) (void)std::stable_sort(first, last);
    if (!TEST_IS_CONSTANT_EVALUATED) (void)std::stable_sort(first, last, std::less<void*>());
    (void)std::swap_ranges(first, last, first2);
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
