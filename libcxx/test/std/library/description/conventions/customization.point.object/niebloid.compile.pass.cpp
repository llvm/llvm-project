//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts, libcpp-has-no-incomplete-ranges
// REQUIRES: stdlib=libc++

// [algorithms.requirements]/2
// [range.iter.ops.general]/2

#include <algorithm>
#include <concepts>
#include <iterator>
#include <memory>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>

// Niebloids, unlike CPOs, are *not* required to be semiregular or even to have
// a declared type at all; they are specified as "magic" overload sets whose
// names are not found by argument-dependent lookup and which inhibit
// argument-dependent lookup if they are found via a `using`-declaration.
//
// libc++ implements them using the same function-object technique we use for CPOs;
// therefore this file should stay in sync with ./cpo.compile.pass.cpp.

template <class CPO, class... Args>
constexpr bool test(CPO& o, Args&&...) {
  static_assert(std::is_const_v<CPO>);
  static_assert(std::is_class_v<CPO>);
  static_assert(std::is_trivial_v<CPO>);

  auto p = o;
  using T = decltype(p);

  // The type of a customization point object, ignoring cv-qualifiers, shall model semiregular.
  static_assert(std::semiregular<T>);

  // The type T of a customization point object, ignoring cv-qualifiers, shall model...
  static_assert(std::invocable<T&, Args...>);
  static_assert(std::invocable<const T&, Args...>);
  static_assert(std::invocable<T, Args...>);
  static_assert(std::invocable<const T, Args...>);

  return true;
}

int *p;
int a[10];
//auto odd = [](int x) { return x % 2 != 0; };
//auto triple = [](int x) { return 3*x; };
//auto plus = [](int x, int y) { return x == y; };
//std::mt19937 g;

// [algorithm.syn]

//static_assert(test(std::ranges::adjacent_find, a));
//static_assert(test(std::ranges::all_of, a, odd));
//static_assert(test(std::ranges::any_of, a, odd));
//static_assert(test(std::ranges::binary_search, a, 42));
//static_assert(test(std::ranges::clamp, 42, 42, 42));
//static_assert(test(std::ranges::copy, a, a));
//static_assert(test(std::ranges::copy_backward, a, a));
//static_assert(test(std::ranges::copy_if, a, a, odd));
//static_assert(test(std::ranges::copy_n, a, 10, a));
//static_assert(test(std::ranges::count, a, 42));
//static_assert(test(std::ranges::count_if, a, odd));
//static_assert(test(std::ranges::ends_with, a, a));
//static_assert(test(std::ranges::equal, a, a));
//static_assert(test(std::ranges::equal_range, a, 42));
//static_assert(test(std::ranges::fill, a, 42));
//static_assert(test(std::ranges::fill_n, a, 10, 42));
//static_assert(test(std::ranges::find, a, 42));
//static_assert(test(std::ranges::find_end, a, a));
//static_assert(test(std::ranges::find_first_of, a, a));
//static_assert(test(std::ranges::find_if, a, odd));
//static_assert(test(std::ranges::find_if_not, a, odd));
//static_assert(test(std::ranges::for_each, a, odd));
//static_assert(test(std::ranges::for_each_n, a, 10, odd));
//static_assert(test(std::ranges::generate, a, 42));
//static_assert(test(std::ranges::generate_n, a, 10, 42));
//static_assert(test(std::ranges::includes, a, a));
//static_assert(test(std::ranges::inplace_merge, a, a+5));
//static_assert(test(std::ranges::is_heap, a));
//static_assert(test(std::ranges::is_heap_until, a));
//static_assert(test(std::ranges::is_partitioned, a, odd));
//static_assert(test(std::ranges::is_permutation, a, a));
//static_assert(test(std::ranges::is_sorted, a));
//static_assert(test(std::ranges::is_sorted_until, a));
//static_assert(test(std::ranges::lexicographical_compare, a, a));
//static_assert(test(std::ranges::lower_bound, a, 42));
//static_assert(test(std::ranges::make_heap, a));
//static_assert(test(std::ranges::max, a));
//static_assert(test(std::ranges::max_element, a));
//static_assert(test(std::ranges::merge, a, a, a));
//static_assert(test(std::ranges::min, a));
//static_assert(test(std::ranges::min_element, a));
//static_assert(test(std::ranges::minmax, a));
//static_assert(test(std::ranges::minmax_element, a));
//static_assert(test(std::ranges::mismatch, a, a));
//static_assert(test(std::ranges::move, a, a));
//static_assert(test(std::ranges::move_backward, a, a));
//static_assert(test(std::ranges::next_permutation, a));
//static_assert(test(std::ranges::none_of, a, odd));
//static_assert(test(std::ranges::nth_element, a, a+5));
//static_assert(test(std::ranges::partial_sort, a, a+5));
//static_assert(test(std::ranges::partial_sort_copy, a, a));
//static_assert(test(std::ranges::partition, a, odd));
//static_assert(test(std::ranges::partition_copy, a, a, a, odd));
//static_assert(test(std::ranges::partition_point, a, odd));
//static_assert(test(std::ranges::pop_heap, a));
//static_assert(test(std::ranges::prev_permutation, a));
//static_assert(test(std::ranges::push_heap, a));
//static_assert(test(std::ranges::remove, a, 42));
//static_assert(test(std::ranges::remove_copy, a, a, 42));
//static_assert(test(std::ranges::remove_copy_if, a, a, odd));
//static_assert(test(std::ranges::remove_if, a, odd));
//static_assert(test(std::ranges::replace, a, 42, 43));
//static_assert(test(std::ranges::replace_copy, a, a, 42, 43));
//static_assert(test(std::ranges::replace_copy_if, a, a, odd, 43));
//static_assert(test(std::ranges::replace_if, a, odd, 43));
//static_assert(test(std::ranges::reverse, a));
//static_assert(test(std::ranges::reverse_copy, a, a));
//static_assert(test(std::ranges::rotate, a, a+5));
//static_assert(test(std::ranges::rotate_copy, a, a+5, a));
//static_assert(test(std::ranges::sample, a, a, 5));
//static_assert(test(std::ranges::search, a, a));
//static_assert(test(std::ranges::search_n, a, 10, 42));
//static_assert(test(std::ranges::set_difference, a, a, a));
//static_assert(test(std::ranges::set_intersection, a, a, a));
//static_assert(test(std::ranges::set_symmetric_difference, a, a, a));
//static_assert(test(std::ranges::set_union, a, a, a));
//static_assert(test(std::ranges::shuffle, a, g));
//static_assert(test(std::ranges::sort, a));
//static_assert(test(std::ranges::sort_heap, a));
//static_assert(test(std::ranges::stable_partition, a, odd));
//static_assert(test(std::ranges::stable_sort, a));
//static_assert(test(std::ranges::starts_with, a, a));
static_assert(test(std::ranges::swap_ranges, a, a));
//static_assert(test(std::ranges::transform, a, a, triple));
//static_assert(test(std::ranges::unique, a));
//static_assert(test(std::ranges::unique_copy, a, a));
//static_assert(test(std::ranges::upper_bound, a, 42));

// [memory.syn]

static_assert(test(std::ranges::construct_at, a, 42));
static_assert(test(std::ranges::destroy, a));
static_assert(test(std::ranges::destroy, a, a+10));
static_assert(test(std::ranges::destroy_at, a));
static_assert(test(std::ranges::destroy_n, a, 10));
static_assert(test(std::ranges::uninitialized_copy, a, a));
static_assert(test(std::ranges::uninitialized_copy, a, a+10, a, a+10));
static_assert(test(std::ranges::uninitialized_copy_n, a, 10, a, a+10));
static_assert(test(std::ranges::uninitialized_default_construct, a));
static_assert(test(std::ranges::uninitialized_default_construct, a, a+10));
static_assert(test(std::ranges::uninitialized_default_construct_n, a, 10));
static_assert(test(std::ranges::uninitialized_fill, a, 42));
static_assert(test(std::ranges::uninitialized_fill, a, a+10, 42));
static_assert(test(std::ranges::uninitialized_fill_n, a, 10, 42));
static_assert(test(std::ranges::uninitialized_move, a, a));
static_assert(test(std::ranges::uninitialized_move, a, a+10, a, a+10));
static_assert(test(std::ranges::uninitialized_move_n, a, 10, a, a+10));
static_assert(test(std::ranges::uninitialized_value_construct, a));
static_assert(test(std::ranges::uninitialized_value_construct, a, a+10));
static_assert(test(std::ranges::uninitialized_value_construct_n, a, 10));

// [numeric.ops.overview] currently has no ranges algorithms. See P1813, P2214

// [range.iter.ops]

static_assert(test(std::ranges::advance, p, 5));
static_assert(test(std::ranges::advance, p, 5, a+10));
static_assert(test(std::ranges::advance, p, a+10));
static_assert(test(std::ranges::distance, a));
static_assert(test(std::ranges::distance, a, a+10));
static_assert(test(std::ranges::next, a));
static_assert(test(std::ranges::next, a, 5));
static_assert(test(std::ranges::next, a, 5, a+10));
static_assert(test(std::ranges::next, a, a+10));
static_assert(test(std::ranges::prev, a+10));
static_assert(test(std::ranges::prev, a+10, 5));
static_assert(test(std::ranges::prev, a+10, 5, a));
