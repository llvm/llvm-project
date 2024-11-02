//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that all STL classic algorithms can be instantiated with a C++20-hostile iterator

// ADDITIONAL_COMPILE_FLAGS: -Wno-ambiguous-reversed-operator

#include <algorithm>
#include <functional>
#include <random>

#include "test_macros.h"

template <class Sub, class Iterator>
struct IteratorAdaptorBase {
  using OutTraits = std::iterator_traits<Iterator>;
  using iterator_category = typename OutTraits::iterator_category;
  using value_type = typename OutTraits::value_type;
  using pointer = typename OutTraits::pointer;
  using reference = typename OutTraits::reference;
  using difference_type = typename OutTraits::difference_type;

  IteratorAdaptorBase() {}
  IteratorAdaptorBase(Iterator) {}

  Sub& sub() { return static_cast<Sub&>(*this); }
  const Sub& sub() const { return static_cast<Sub&>(*this); }

  const Iterator& base() const { return it_; }

  reference get() const { return *it_; }
  reference operator*() const { return *it_; }
  pointer operator->() const { return it_; }
  reference operator[](difference_type) const { return *it_; }

  Sub& operator++() { return static_cast<Sub&>(*this); }
  Sub& operator--() { return static_cast<Sub&>(*this); }
  Sub operator++(int) { return static_cast<Sub&>(*this); }
  Sub operator--(int) { return static_cast<Sub&>(*this); }

  Sub& operator+=(difference_type) { return static_cast<Sub&>(*this); }
  Sub& operator-=(difference_type) { return static_cast<Sub&>(*this); }
  bool operator==(Sub) const { return false; }
  bool operator!=(Sub) const { return false; }
  bool operator==(Iterator b) const { return *this == Sub(b); }
  bool operator!=(Iterator b) const { return *this != Sub(b); }

  friend Sub operator+(Sub, difference_type) { return Sub(); }
  friend Sub operator+(difference_type, Sub) { return Sub(); }
  friend Sub operator-(Sub, difference_type) { return Sub(); }
  friend difference_type operator-(Sub, Sub) { return 0; }

  friend bool operator<(Sub, Sub) { return false; }
  friend bool operator>(Sub, Sub) { return false; }
  friend bool operator<=(Sub, Sub) { return false; }
  friend bool operator>=(Sub, Sub) { return false; }

 private:
  Iterator it_;
};

template <typename It>
struct Cpp20HostileIterator
    : IteratorAdaptorBase<Cpp20HostileIterator<It>, It> {
  Cpp20HostileIterator() {}
  Cpp20HostileIterator(It) {}
};

struct Pred {
  bool operator()(int, int) { return false; }
  bool operator()(int) { return false; }
  int operator()() { return 0; }
};

void test() {
  Cpp20HostileIterator<int*> it;
  Pred pred;
  std::mt19937_64 rng;

  (void) std::adjacent_find(it, it);
  (void) std::adjacent_find(it, it, pred);
  (void) std::all_of(it, it, pred);
  (void) std::any_of(it, it, pred);
  (void) std::binary_search(it, it, 0);
  (void) std::binary_search(it, it, 0, pred);
  (void) std::copy_backward(it, it, it);
  (void) std::copy_if(it, it, it, pred);
  (void) std::copy_n(it, 0, it);
  (void) std::copy(it, it, it);
  (void) std::count_if(it, it, pred);
  (void) std::count(it, it, 0);
  (void) std::equal_range(it, it, 0);
  (void) std::equal_range(it, it, 0, pred);
  (void) std::equal(it, it, it);
  (void) std::equal(it, it, it, pred);
#if TEST_STD_VER > 11
  (void) std::equal(it, it, it, it);
  (void) std::equal(it, it, it, it, pred);
#endif
  (void) std::fill_n(it, 0, 0);
  (void) std::fill(it, it, 0);
  (void) std::find_end(it, it, it, it);
  (void) std::find_end(it, it, it, it, pred);
  (void) std::find_first_of(it, it, it, it);
  (void) std::find_first_of(it, it, it, it, pred);
  (void) std::find_if_not(it, it, pred);
  (void) std::find_if(it, it, pred);
  (void) std::find(it, it, 0);
#if TEST_STD_VER > 14
  (void) std::for_each_n(it, 0, pred);
#endif
  (void) std::for_each(it, it, pred);
  (void) std::generate_n(it, 0, pred);
  (void) std::generate(it, it, pred);
  (void) std::includes(it, it, it, it);
  (void) std::includes(it, it, it, it, pred);
  (void) std::inplace_merge(it, it, it);
  (void) std::inplace_merge(it, it, it, pred);
  (void) std::is_heap_until(it, it);
  (void) std::is_heap_until(it, it, pred);
  (void) std::is_heap(it, it);
  (void) std::is_heap(it, it, pred);
  (void) std::is_partitioned(it, it, pred);
  (void) std::is_permutation(it, it, it);
  (void) std::is_permutation(it, it, it, pred);
#if TEST_STD_VER > 11
  (void) std::is_permutation(it, it, it, it);
  (void) std::is_permutation(it, it, it, it, pred);
#endif
  (void) std::is_sorted_until(it, it);
  (void) std::is_sorted_until(it, it, pred);
  (void) std::is_sorted(it, it);
  (void) std::is_sorted(it, it, pred);
  (void) std::lexicographical_compare(it, it, it, it);
  (void) std::lexicographical_compare(it, it, it, it, pred);
  (void) std::lower_bound(it, it, 0);
  (void) std::lower_bound(it, it, 0, pred);
  (void) std::make_heap(it, it);
  (void) std::make_heap(it, it, pred);
  (void) std::max_element(it, it);
  (void) std::max_element(it, it, pred);
  (void) std::merge(it, it, it, it, it);
  (void) std::merge(it, it, it, it, it, pred);
  (void) std::min_element(it, it);
  (void) std::min_element(it, it, pred);
  (void) std::minmax_element(it, it);
  (void) std::minmax_element(it, it, pred);
  (void) std::mismatch(it, it, it);
  (void) std::mismatch(it, it, it, pred);
  (void) std::move_backward(it, it, it);
  (void) std::move(it, it, it);
  (void) std::next_permutation(it, it);
  (void) std::next_permutation(it, it, pred);
  (void) std::none_of(it, it, pred);
  (void) std::nth_element(it, it, it);
  (void) std::nth_element(it, it, it, pred);
  (void) std::partial_sort_copy(it, it, it, it);
  (void) std::partial_sort_copy(it, it, it, it, pred);
  (void) std::partial_sort(it, it, it);
  (void) std::partial_sort(it, it, it, pred);
  (void) std::partition_copy(it, it, it, it, pred);
  (void) std::partition_point(it, it, pred);
  (void) std::partition(it, it, pred);
  (void) std::pop_heap(it, it);
  (void) std::pop_heap(it, it, pred);
  (void) std::prev_permutation(it, it);
  (void) std::prev_permutation(it, it, pred);
  (void) std::push_heap(it, it);
  (void) std::push_heap(it, it, pred);
  (void) std::remove_copy_if(it, it, it, pred);
  (void) std::remove_copy(it, it, it, 0);
  (void) std::remove_if(it, it, pred);
  (void) std::remove(it, it, 0);
  (void) std::replace_copy_if(it, it, it, pred, 0);
  (void) std::replace_copy(it, it, it, 0, 0);
  (void) std::replace_if(it, it, pred, 0);
  (void) std::replace(it, it, 0, 0);
  (void) std::reverse_copy(it, it, it);
  (void) std::reverse(it, it);
  (void) std::rotate_copy(it, it, it, it);
  (void) std::rotate(it, it, it);
#if TEST_STD_VER > 14
  (void) std::sample(it, it, it, 0, rng);
#endif
  (void) std::search(it, it, it, it);
  (void) std::search(it, it, it, it, pred);
#if TEST_STD_VER > 14
  (void) std::search(it, it, std::default_searcher<Cpp20HostileIterator<int*>>(it, it));
#endif
  (void) std::set_difference(it, it, it, it, it);
  (void) std::set_difference(it, it, it, it, it, pred);
  (void) std::set_intersection(it, it, it, it, it);
  (void) std::set_intersection(it, it, it, it, it, pred);
  (void) std::set_symmetric_difference(it, it, it, it, it);
  (void) std::set_symmetric_difference(it, it, it, it, it, pred);
  (void) std::set_union(it, it, it, it, it);
  (void) std::set_union(it, it, it, it, it, pred);
#if TEST_STD_VER > 17
  (void) std::shift_left(it, it, 0);
  (void) std::shift_right(it, it, 0);
#endif
  (void) std::shuffle(it, it, rng);
  (void) std::sort_heap(it, it);
  (void) std::sort_heap(it, it, pred);
  (void) std::sort(it, it);
  (void) std::sort(it, it, pred);
  (void) std::stable_partition(it, it, pred);
  (void) std::stable_sort(it, it);
  (void) std::swap_ranges(it, it, it);
  (void) std::transform(it, it, it, pred);
  (void) std::transform(it, it, it, it, pred);
  (void) std::unique_copy(it, it, it);
  (void) std::unique(it, it);
  (void) std::upper_bound(it, it, 0);
}
