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
  ThrowOnMove(const ThrowOnMove&) _NOEXCEPT_(false);
  ThrowOnMove(ThrowOnMove&&) _NOEXCEPT_(false);
  ThrowOnMove& operator=(ThrowOnMove&&) _NOEXCEPT_(false);
  ThrowOnMove& operator=(const ThrowOnMove&) _NOEXCEPT_(false);

  bool operator<(ThrowOnMove const&) const;
  bool operator==(ThrowOnMove const&) const;
};

struct NonCopyThrowOnMove {
  NonCopyThrowOnMove();
  NonCopyThrowOnMove(ThrowOnMove&&) _NOEXCEPT_(false);
  NonCopyThrowOnMove(const NonCopyThrowOnMove&) = delete;
  NonCopyThrowOnMove& operator=(ThrowOnMove&&) _NOEXCEPT_(false);
  NonCopyThrowOnMove& operator=(const NonCopyThrowOnMove&) = delete;

  bool operator<(NonCopyThrowOnMove const&) const;
  bool operator==(NonCopyThrowOnMove const&) const;
};

struct ThrowingHash {
  template <class T>
  std::size_t operator()(const T&) const _NOEXCEPT_(false);
};

struct NoThrowHash {
  template <class T>
  std::size_t operator()(const T&) const _NOEXCEPT;
};

template <class T, bool Expected>
void test_emplacement_strong_exception() {
  static_assert(std::__container_traits<T>::__emplacement_has_strong_exception_safety_guarantee::value == Expected, "");
}

void test() {
  test_emplacement_strong_exception<std::list<int>, true>();
  test_emplacement_strong_exception<std::list<int, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::list<MoveOnly>, true>();
  test_emplacement_strong_exception<std::list<ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::list<NonCopyThrowOnMove>, true>();

  test_emplacement_strong_exception<std::forward_list<int>, true>();
  test_emplacement_strong_exception<std::forward_list<int, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::forward_list<MoveOnly>, true>();
  test_emplacement_strong_exception<std::forward_list<ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::forward_list<NonCopyThrowOnMove>, true>();

  test_emplacement_strong_exception<std::deque<int>, true>();
  test_emplacement_strong_exception<std::deque<int, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::deque<MoveOnly>, true>();
  test_emplacement_strong_exception<std::deque<ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::deque<NonCopyThrowOnMove>, false>();

  test_emplacement_strong_exception<std::vector<int>, true>();
  test_emplacement_strong_exception<std::vector<int, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::vector<MoveOnly>, true>();
  test_emplacement_strong_exception<std::vector<ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::vector<NonCopyThrowOnMove>, false>();

  test_emplacement_strong_exception<std::set<int>, true>();
  test_emplacement_strong_exception<std::set<int, std::less<int>, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::set<MoveOnly>, true>();
  test_emplacement_strong_exception<std::set<ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::set<NonCopyThrowOnMove>, true>();

  test_emplacement_strong_exception<std::multiset<int>, true>();
  test_emplacement_strong_exception<std::multiset<int, std::less<int>, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::multiset<MoveOnly>, true>();
  test_emplacement_strong_exception<std::multiset<ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::multiset<NonCopyThrowOnMove>, true>();

  test_emplacement_strong_exception<std::map<int, int>, true>();
  test_emplacement_strong_exception<std::map<int, int, std::less<int>, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::map<MoveOnly, MoveOnly>, true>();
  test_emplacement_strong_exception<std::map<ThrowOnMove, ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::map<NonCopyThrowOnMove, NonCopyThrowOnMove>, true>();

  test_emplacement_strong_exception<std::multimap<int, int>, true>();
  test_emplacement_strong_exception<std::multimap<int, int, std::less<int>, test_allocator<int> >, true>();
  test_emplacement_strong_exception<std::multimap<MoveOnly, MoveOnly>, true>();
  test_emplacement_strong_exception<std::multimap<ThrowOnMove, ThrowOnMove>, true>();
  test_emplacement_strong_exception<std::multimap<NonCopyThrowOnMove, NonCopyThrowOnMove>, true>();

#if TEST_STD_VER < 11
  test_emplacement_strong_exception<std::unordered_set<int>, false>();
  test_emplacement_strong_exception<std::unordered_set<int, std::hash<int>, std::less<int>, test_allocator<int> >,
                                    false>();
  test_emplacement_strong_exception<std::unordered_set<MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_set<MoveOnly, NoThrowHash>, false>();
  test_emplacement_strong_exception<std::unordered_set<MoveOnly, ThrowingHash>, false>();

  test_emplacement_strong_exception<std::unordered_multiset<int>, false>();
  test_emplacement_strong_exception<std::unordered_multiset<int, std::hash<int>, std::less<int>, test_allocator<int> >,
                                    false>();
  test_emplacement_strong_exception<std::unordered_multiset<MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_multiset<MoveOnly, NoThrowHash>, false>();
  test_emplacement_strong_exception<std::unordered_multiset<MoveOnly, ThrowingHash>, false>();

  test_emplacement_strong_exception<std::unordered_map<int, int>, false>();
  test_emplacement_strong_exception<std::unordered_map<int, int, std::hash<int>, std::less<int>, test_allocator<int> >,
                                    false>();
  test_emplacement_strong_exception<std::unordered_map<MoveOnly, MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_map<MoveOnly, MoveOnly, NoThrowHash>, false>();
  test_emplacement_strong_exception<std::unordered_map<MoveOnly, MoveOnly, ThrowingHash>, false>();

  test_emplacement_strong_exception<std::unordered_multimap<int, int>, false>();
  test_emplacement_strong_exception<
      std::unordered_multimap<int, int, std::hash<int>, std::less<int>, test_allocator<int> >,
      false>();
  test_emplacement_strong_exception<std::unordered_multimap<MoveOnly, MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_multimap<MoveOnly, MoveOnly, NoThrowHash>, false>();
  test_emplacement_strong_exception<std::unordered_multimap<MoveOnly, MoveOnly, ThrowingHash>, false>();
#else

  test_emplacement_strong_exception<std::unordered_set<int>, true>();
  test_emplacement_strong_exception<std::unordered_set<int, std::hash<int>, std::less<int>, test_allocator<int> >,
                                    true>();
  test_emplacement_strong_exception<std::unordered_set<MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_set<MoveOnly, NoThrowHash>, true>();
  test_emplacement_strong_exception<std::unordered_set<MoveOnly, ThrowingHash>, false>();

  test_emplacement_strong_exception<std::unordered_multiset<int>, true>();
  test_emplacement_strong_exception<std::unordered_multiset<int, std::hash<int>, std::less<int>, test_allocator<int> >,
                                    true>();
  test_emplacement_strong_exception<std::unordered_multiset<MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_multiset<MoveOnly, NoThrowHash>, true>();
  test_emplacement_strong_exception<std::unordered_multiset<MoveOnly, ThrowingHash>, false>();

  test_emplacement_strong_exception<std::unordered_map<int, int>, true>();
  test_emplacement_strong_exception<std::unordered_map<int, int, std::hash<int>, std::less<int>, test_allocator<int> >,
                                    true>();
  test_emplacement_strong_exception<std::unordered_map<MoveOnly, MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_map<MoveOnly, MoveOnly, NoThrowHash>, true>();
  test_emplacement_strong_exception<std::unordered_map<MoveOnly, MoveOnly, ThrowingHash>, false>();

  test_emplacement_strong_exception<std::unordered_multimap<int, int>, true>();
  test_emplacement_strong_exception<
      std::unordered_multimap<int, int, std::hash<int>, std::less<int>, test_allocator<int> >,
      true>();
  test_emplacement_strong_exception<std::unordered_multimap<MoveOnly, MoveOnly>, false>();
  test_emplacement_strong_exception<std::unordered_multimap<MoveOnly, MoveOnly, NoThrowHash>, true>();
  test_emplacement_strong_exception<std::unordered_multimap<MoveOnly, MoveOnly, ThrowingHash>, false>();
#endif
}
