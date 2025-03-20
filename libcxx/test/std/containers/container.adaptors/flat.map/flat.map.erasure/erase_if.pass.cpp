//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class Key, class T, class Compare, class KeyContainer, class MappedContainer, class Predicate>
//   typename flat_map<Key, T, Compare, KeyContainer, MappedContainer>::size_type
//   erase_if(flat_map<Key, T, Compare, KeyContainer, MappedContainer>& c, Predicate pred);

#include <deque>
#include <flat_map>
#include <functional>
#include <initializer_list>
#include <vector>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

// Verify that `flat_map` (like `map`) does NOT support std::erase.
//
template <class S>
concept HasStdErase = requires(S& s, typename S::value_type x) { std::erase(s, x); };
static_assert(HasStdErase<std::vector<int>>);
static_assert(!HasStdErase<std::flat_map<int, int>>);

template <class M>
M make(std::initializer_list<int> vals) {
  M ret;
  for (int v : vals)
    ret[static_cast<typename M::key_type>(v)] = static_cast<typename M::mapped_type>(v + 10);
  return ret;
}

template <class M, class Pred>
void test0(
    std::initializer_list<int> vals, Pred p, std::initializer_list<int> expected, std::size_t expected_erased_count) {
  M s = make<M>(vals);
  ASSERT_SAME_TYPE(typename M::size_type, decltype(std::erase_if(s, p)));
  assert(expected_erased_count == std::erase_if(s, p));
  assert(s == make<M>(expected));
}

template <class S>
void test() {
  // Test all the plausible signatures for this predicate.
  auto is1   = [](typename S::const_reference v) { return v.first == 1; };
  auto is2   = [](typename S::value_type v) { return v.first == 2; };
  auto is3   = [](const typename S::value_type& v) { return v.first == 3; };
  auto is4   = [](auto v) { return v.first == 4; };
  auto True  = [](const auto&) { return true; };
  auto False = [](auto&&) { return false; };

  test0<S>({}, is1, {}, 0);

  test0<S>({1}, is1, {}, 1);
  test0<S>({1}, is2, {1}, 0);

  test0<S>({1, 2}, is1, {2}, 1);
  test0<S>({1, 2}, is2, {1}, 1);
  test0<S>({1, 2}, is3, {1, 2}, 0);

  test0<S>({1, 2, 3}, is1, {2, 3}, 1);
  test0<S>({1, 2, 3}, is2, {1, 3}, 1);
  test0<S>({1, 2, 3}, is3, {1, 2}, 1);
  test0<S>({1, 2, 3}, is4, {1, 2, 3}, 0);

  test0<S>({1, 2, 3}, True, {}, 3);
  test0<S>({1, 2, 3}, False, {1, 2, 3}, 0);
}

int main(int, char**) {
  test<std::flat_map<int, char>>();
  test<std::flat_map<int,
                     char,
                     std::less<int>,
                     std::vector<int, min_allocator<int>>,
                     std::vector<char, min_allocator<char>>>>();
  test<std::flat_map<int, char, std::greater<int>, std::vector<int, test_allocator<int>>>>();
  test<std::flat_map<int, char, std::less<int>, std::deque<int, min_allocator<int>>>>();
  test<std::flat_map<int, char, std::greater<int>, std::deque<int, test_allocator<int>>>>();
  test<std::flat_map<long, int>>();
  test<std::flat_map<double, int>>();

  return 0;
}
