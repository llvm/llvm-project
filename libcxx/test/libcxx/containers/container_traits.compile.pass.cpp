//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <__type_traits/container_traits.h>
//

#include <__type_traits/container_traits.h>

#include <deque>
#include <forward_list>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "test_allocator.h"
#include "test_macros.h"
#include "MoveOnly.h"

struct ThrowOnMove {
  ThrowOnMove();
  ThrowOnMove(const ThrowOnMove&) TEST_NOEXCEPT_COND(false);
  ThrowOnMove(ThrowOnMove&&) TEST_NOEXCEPT_COND(false);
  ThrowOnMove& operator=(ThrowOnMove&&) TEST_NOEXCEPT_COND(false);
  ThrowOnMove& operator=(const ThrowOnMove&) TEST_NOEXCEPT_COND(false);

  bool operator<(ThrowOnMove const&) const;
  bool operator==(ThrowOnMove const&) const;
};

struct NonCopyThrowOnMove {
  NonCopyThrowOnMove();
  NonCopyThrowOnMove(NonCopyThrowOnMove&&) TEST_NOEXCEPT_COND(false);
  NonCopyThrowOnMove(const NonCopyThrowOnMove&) = delete;
  NonCopyThrowOnMove& operator=(NonCopyThrowOnMove&&) TEST_NOEXCEPT_COND(false);
  NonCopyThrowOnMove& operator=(const NonCopyThrowOnMove&) = delete;

  bool operator<(NonCopyThrowOnMove const&) const;
  bool operator==(NonCopyThrowOnMove const&) const;
};

struct ThrowingHash {
  template <class T>
  std::size_t operator()(const T&) const TEST_NOEXCEPT_COND(false);
};

struct NoThrowHash {
  template <class T>
  std::size_t operator()(const T&) const TEST_NOEXCEPT;
};

template <bool Expected, class Container>
void check() {
  static_assert(
      std::__container_traits<Container>::__emplacement_has_strong_exception_safety_guarantee == Expected, "");
}

void test() {
  check<true, std::list<int> >();
  check<true, std::list<int, test_allocator<int> > >();
  check<true, std::list<MoveOnly> >();
  check<true, std::list<ThrowOnMove> >();
  check<true, std::list<NonCopyThrowOnMove> >();

  check<true, std::forward_list<int> >();
  check<true, std::forward_list<int, test_allocator<int> > >();
  check<true, std::forward_list<MoveOnly> >();
  check<true, std::forward_list<ThrowOnMove> >();
  check<true, std::forward_list<NonCopyThrowOnMove> >();

  check<true, std::deque<int> >();
  check<true, std::deque<int, test_allocator<int> > >();
  check<true, std::deque<MoveOnly> >();
  check<true, std::deque<ThrowOnMove> >();
  check<false, std::deque<NonCopyThrowOnMove> >();

  check<true, std::vector<int> >();
  check<true, std::vector<int, test_allocator<int> > >();
  check<true, std::vector<MoveOnly> >();
  check<true, std::vector<ThrowOnMove> >();
  check<false, std::vector<NonCopyThrowOnMove> >();

  check<true, std::set<int> >();
  check<true, std::set<int, std::less<int>, test_allocator<int> > >();
  check<true, std::set<MoveOnly> >();
  check<true, std::set<ThrowOnMove> >();
  check<true, std::set<NonCopyThrowOnMove> >();

  check<true, std::multiset<int> >();
  check<true, std::multiset<int, std::less<int>, test_allocator<int> > >();
  check<true, std::multiset<MoveOnly> >();
  check<true, std::multiset<ThrowOnMove> >();
  check<true, std::multiset<NonCopyThrowOnMove> >();

  check<true, std::map<int, int> >();
  check<true, std::map<int, int, std::less<int>, test_allocator<int> > >();
  check<true, std::map<MoveOnly, MoveOnly> >();
  check<true, std::map<ThrowOnMove, ThrowOnMove> >();
  check<true, std::map<NonCopyThrowOnMove, NonCopyThrowOnMove> >();

  check<true, std::multimap<int, int> >();
  check<true, std::multimap<int, int, std::less<int>, test_allocator<int> > >();
  check<true, std::multimap<MoveOnly, MoveOnly> >();
  check<true, std::multimap<ThrowOnMove, ThrowOnMove> >();
  check<true, std::multimap<NonCopyThrowOnMove, NonCopyThrowOnMove> >();

#if TEST_STD_VER < 11
  check<false, std::unordered_set<int> >();
  check<false, std::unordered_set<int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_set<MoveOnly> >();
  check<false, std::unordered_set<MoveOnly, NoThrowHash> >();
  check<false, std::unordered_set<MoveOnly, ThrowingHash> >();

  check<false, std::unordered_multiset<int> >();
  check<false, std::unordered_multiset<int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_multiset<MoveOnly> >();
  check<false, std::unordered_multiset<MoveOnly, NoThrowHash> >();
  check<false, std::unordered_multiset<MoveOnly, ThrowingHash> >();

  check<false, std::unordered_map<int, int> >();
  check<false, std::unordered_map<int, int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_map<MoveOnly, MoveOnly> >();
  check<false, std::unordered_map<MoveOnly, MoveOnly, NoThrowHash> >();
  check<false, std::unordered_map<MoveOnly, MoveOnly, ThrowingHash> >();

  check<false, std::unordered_multimap<int, int> >();
  check<false, std::unordered_multimap<int, int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_multimap<MoveOnly, MoveOnly> >();
  check<false, std::unordered_multimap<MoveOnly, MoveOnly, NoThrowHash> >();
  check<false, std::unordered_multimap<MoveOnly, MoveOnly, ThrowingHash> >();
#else
  check<true, std::unordered_set<int> >();
  check<true, std::unordered_set<int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_set<MoveOnly> >();
  check<true, std::unordered_set<MoveOnly, NoThrowHash> >();
  check<false, std::unordered_set<MoveOnly, ThrowingHash> >();

  check<true, std::unordered_multiset<int> >();
  check<true, std::unordered_multiset<int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_multiset<MoveOnly> >();
  check<true, std::unordered_multiset<MoveOnly, NoThrowHash> >();
  check<false, std::unordered_multiset<MoveOnly, ThrowingHash> >();

  check<true, std::unordered_map<int, int> >();
  check<true, std::unordered_map<int, int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_map<MoveOnly, MoveOnly> >();
  check<true, std::unordered_map<MoveOnly, MoveOnly, NoThrowHash> >();
  check<false, std::unordered_map<MoveOnly, MoveOnly, ThrowingHash> >();

  check<true, std::unordered_multimap<int, int> >();
  check<true, std::unordered_multimap<int, int, std::hash<int>, std::less<int>, test_allocator<int> > >();
  check<false, std::unordered_multimap<MoveOnly, MoveOnly> >();
  check<true, std::unordered_multimap<MoveOnly, MoveOnly, NoThrowHash> >();
  check<false, std::unordered_multimap<MoveOnly, MoveOnly, ThrowingHash> >();
#endif
}
