//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

#include <algorithm>
#include <functional>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };

template<class>
struct Intable {
    TEST_CONSTEXPR operator int() const { return 1; }
};

struct Tester {
    using Element = Holder<Incomplete>*;
    Element data[10];
};

TEST_CONSTEXPR_CXX20 bool test()
{
    Tester t {};
    Tester u {};
    Tester::Element value = nullptr;
    Intable<Tester::Element> count;

    // THESE RELY ON ADL SWAP IN PRACTICE:
    // swap_ranges, iter_swap, reverse, rotate, partition
    // sort, nth_element
    // pop_heap, sort_heap, partial_sort, partial_sort_copy
    // next_permutation, prev_permutation
    // stable_partition, stable_sort, inplace_merge
    // THESE RELY ON ADL SWAP IN THEORY:
    // push_heap, make_heap

    (void)std::all_of(t.data, t.data+10, [](void*){ return true; });
    (void)std::any_of(t.data, t.data+10, [](void*){ return true; });
    (void)std::copy(t.data, t.data+10, u.data);
    (void)std::copy_n(t.data, count, u.data);
    (void)std::copy_backward(t.data, t.data+10, u.data+10);
    (void)std::count(t.data, t.data+10, value);
    (void)std::count_if(t.data, t.data+10, [](void*){ return true; });
    (void)std::distance(t.data, t.data+10);
    (void)std::fill(t.data, t.data+10, value);
    (void)std::fill_n(t.data, count, value);
    (void)std::find_if(t.data, t.data+10, [](void*){ return true; });
    (void)std::find_if_not(t.data, t.data+10, [](void*){ return true; });
    (void)std::for_each(t.data, t.data+10, [](void*){});
#if TEST_STD_VER >= 17
    (void)std::for_each_n(t.data, count, [](void*){});
#endif
    (void)std::generate(t.data, t.data+10, [](){ return nullptr; });
    (void)std::generate_n(t.data, count, [](){ return nullptr; });
    (void)std::is_partitioned(t.data, t.data+10, [](void*){ return true; });
    (void)std::move(t.data, t.data+10, u.data);
    (void)std::move_backward(t.data, t.data+10, u.data+10);
    (void)std::none_of(t.data, t.data+10, [](void*){ return true; });
    (void)std::partition_copy(t.data, t.data+5, u.data, u.data+5, [](void*){ return true; });
    (void)std::partition_point(t.data, t.data+10, [](void*){ return true; });
    (void)std::remove(t.data, t.data+10, value);
    (void)std::remove_copy(t.data, t.data+10, u.data, value);
    (void)std::remove_copy_if(t.data, t.data+10, u.data, [](void*){ return true; });
    (void)std::remove_if(t.data, t.data+10, [](void*){ return true; });
    (void)std::replace(t.data, t.data+10, value, value);
    (void)std::replace_copy(t.data, t.data+10, u.data, value, value);
    (void)std::replace_copy_if(t.data, t.data+10, u.data, [](void*){ return true; }, value);
    (void)std::replace_if(t.data, t.data+10, [](void*){ return true; }, value);
    (void)std::reverse_copy(t.data, t.data+10, u.data);
    (void)std::rotate_copy(t.data, t.data+5, t.data+10, u.data);
    // TODO: shift_left
    // TODO: shift_right
    (void)std::transform(t.data, t.data+10, u.data, [](void*){ return nullptr; });

    // WITHOUT COMPARATORS
    (void)std::adjacent_find(t.data, t.data+10);
    (void)std::binary_search(t.data, t.data+10, t.data[5]);
    (void)std::equal(t.data, t.data+10, u.data);
    (void)std::equal_range(t.data, t.data+10, t.data[5]);
    (void)std::find_end(t.data, t.data+10, u.data, u.data+5);
    (void)std::includes(t.data, t.data+10, u.data, u.data+10);
    (void)std::is_heap(t.data, t.data+10);
    (void)std::is_heap_until(t.data, t.data+10);
    (void)std::is_permutation(t.data, t.data+10, u.data);
    (void)std::is_sorted(t.data, t.data+10);
    (void)std::is_sorted_until(t.data, t.data+10);
    (void)std::lexicographical_compare(t.data, t.data+10, u.data, u.data+10);
    // TODO: lexicographical_compare_three_way
    (void)std::lower_bound(t.data, t.data+10, t.data[5]);
    (void)std::max(value, value);
    (void)std::max({ value, value });
    (void)std::max_element(t.data, t.data+10);
    (void)std::merge(t.data, t.data+5, t.data+5, t.data+10, u.data);
    (void)std::min(value, value);
    (void)std::min({ value, value });
    (void)std::min_element(t.data, t.data+10);
    (void)std::minmax(value, value);
    (void)std::minmax({ value, value });
    (void)std::minmax_element(t.data, t.data+10);
    (void)std::mismatch(t.data, t.data+10, u.data);
    (void)std::search(t.data, t.data+10, u.data, u.data+5);
    (void)std::search_n(t.data, t.data+10, count, value);
    (void)std::set_difference(t.data, t.data+5, t.data+5, t.data+10, u.data);
    (void)std::set_intersection(t.data, t.data+5, t.data+5, t.data+10, u.data);
    (void)std::set_symmetric_difference(t.data, t.data+5, t.data+5, t.data+10, u.data);
    (void)std::set_union(t.data, t.data+5, t.data+5, t.data+10, u.data);
    (void)std::unique(t.data, t.data+10);
    (void)std::unique_copy(t.data, t.data+10, u.data);
    (void)std::upper_bound(t.data, t.data+10, t.data[5]);
#if TEST_STD_VER >= 14
    (void)std::equal(t.data, t.data+10, u.data, u.data+10);
    (void)std::is_permutation(t.data, t.data+10, u.data, u.data+10);
    (void)std::mismatch(t.data, t.data+10, u.data, u.data+10);
#endif
#if TEST_STD_VER >= 20
    (void)std::clamp(value, value, value);
#endif

    // WITH COMPARATORS
    (void)std::adjacent_find(t.data, t.data+10, std::equal_to<void*>());
    (void)std::binary_search(t.data, t.data+10, value, std::less<void*>());
    (void)std::equal(t.data, t.data+10, u.data, std::equal_to<void*>());
    (void)std::equal_range(t.data, t.data+10, value, std::less<void*>());
    (void)std::find_end(t.data, t.data+10, u.data, u.data+5, std::equal_to<void*>());
    (void)std::includes(t.data, t.data+10, u.data, u.data+10, std::less<void*>());
    (void)std::is_heap(t.data, t.data+10, std::less<void*>());
    (void)std::is_heap_until(t.data, t.data+10, std::less<void*>());
    (void)std::is_permutation(t.data, t.data+10, u.data, std::equal_to<void*>());
    (void)std::is_sorted(t.data, t.data+10, std::less<void*>());
    (void)std::is_sorted_until(t.data, t.data+10, std::less<void*>());
    (void)std::lexicographical_compare(t.data, t.data+10, u.data, u.data+10, std::less<void*>());
    // TODO: lexicographical_compare_three_way
    (void)std::lower_bound(t.data, t.data+10, value, std::less<void*>());
    (void)std::max(value, value, std::less<void*>());
    (void)std::max({ value, value }, std::less<void*>());
    (void)std::max_element(t.data, t.data+10, std::less<void*>());
    (void)std::merge(t.data, t.data+5, t.data+5, t.data+10, u.data, std::less<void*>());
    (void)std::min(value, value, std::less<void*>());
    (void)std::min({ value, value }, std::less<void*>());
    (void)std::min_element(t.data, t.data+10, std::less<void*>());
    (void)std::minmax(value, value, std::less<void*>());
    (void)std::minmax({ value, value }, std::less<void*>());
    (void)std::minmax_element(t.data, t.data+10, std::less<void*>());
    (void)std::mismatch(t.data, t.data+10, u.data, std::equal_to<void*>());
    (void)std::search(t.data, t.data+10, u.data, u.data+5, std::equal_to<void*>());
    (void)std::search_n(t.data, t.data+10, count, value, std::equal_to<void*>());
    (void)std::set_difference(t.data, t.data+5, t.data+5, t.data+10, u.data, std::less<void*>());
    (void)std::set_intersection(t.data, t.data+5, t.data+5, t.data+10, u.data, std::less<void*>());
    (void)std::set_symmetric_difference(t.data, t.data+5, t.data+5, t.data+10, u.data, std::less<void*>());
    (void)std::set_union(t.data, t.data+5, t.data+5, t.data+10, u.data, std::less<void*>());
    (void)std::unique(t.data, t.data+10, std::equal_to<void*>());
    (void)std::unique_copy(t.data, t.data+10, u.data, std::equal_to<void*>());
    (void)std::upper_bound(t.data, t.data+10, value, std::less<void*>());
#if TEST_STD_VER >= 14
    (void)std::equal(t.data, t.data+10, u.data, u.data+10, std::equal_to<void*>());
    (void)std::is_permutation(t.data, t.data+10, u.data, u.data+10, std::equal_to<void*>());
    (void)std::mismatch(t.data, t.data+10, u.data, u.data+10, std::equal_to<void*>());
#endif
#if TEST_STD_VER >= 20
    (void)std::clamp(value, value, value, std::less<void*>());
#endif

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif
    return 0;
}
