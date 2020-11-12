//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TRANSPARENT_UNORDERED_H
#define TEST_TRANSPARENT_UNORDERED_H

#include "test_macros.h"
#include "is_transparent.h"

#include <cassert>

// testing transparent unordered containers
#if TEST_STD_VER > 17

template <template <typename...> class UnorderedSet, typename Hash,
          typename Equal>
using unord_set_type = UnorderedSet<StoredType<int>, Hash, Equal>;

template <template <typename...> class UnorderedMap, typename Hash,
          typename Equal>
using unord_map_type = UnorderedMap<StoredType<int>, int, Hash, Equal>;

template <typename Container, typename... Args>
void test_transparent_find(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  assert(c.find(SearchedType<int>(1, &conversions)) != c.end());
  assert(c.find(SearchedType<int>(2, &conversions)) != c.end());
  assert(conversions == 0);
  assert(c.find(SearchedType<int>(3, &conversions)) == c.end());
  assert(conversions == 0);
}

template <typename Container, typename... Args>
void test_non_transparent_find(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  assert(c.find(SearchedType<int>(1, &conversions)) != c.end());
  assert(conversions > 0);
  conversions = 0;
  assert(c.find(SearchedType<int>(2, &conversions)) != c.end());
  assert(conversions > 0);
  conversions = 0;
  assert(c.find(SearchedType<int>(3, &conversions)) == c.end());
  assert(conversions > 0);
}

template <typename Container, typename... Args>
void test_transparent_count(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  assert(c.count(SearchedType<int>(1, &conversions)) > 0);
  assert(c.count(SearchedType<int>(2, &conversions)) > 0);
  assert(conversions == 0);
  assert(c.count(SearchedType<int>(3, &conversions)) == 0);
  assert(conversions == 0);
}

template <typename Container, typename... Args>
void test_non_transparent_count(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  assert(c.count(SearchedType<int>(1, &conversions)) > 0);
  assert(conversions > 0);
  conversions = 0;
  assert(c.count(SearchedType<int>(2, &conversions)) > 0);
  assert(conversions > 0);
  conversions = 0;
  assert(c.count(SearchedType<int>(3, &conversions)) == 0);
  assert(conversions > 0);
}

template <typename Container, typename... Args>
void test_transparent_contains(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  assert(c.contains(SearchedType<int>(1, &conversions)));
  assert(c.contains(SearchedType<int>(2, &conversions)));
  assert(conversions == 0);
  assert(!c.contains(SearchedType<int>(3, &conversions)));
  assert(conversions == 0);
}

template <typename Container, typename... Args>
void test_non_transparent_contains(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  assert(c.contains(SearchedType<int>(1, &conversions)));
  assert(conversions > 0);
  conversions = 0;
  assert(c.contains(SearchedType<int>(2, &conversions)));
  assert(conversions > 0);
  conversions = 0;
  assert(!c.contains(SearchedType<int>(3, &conversions)));
  assert(conversions > 0);
}

template <typename Container, typename... Args>
void test_transparent_equal_range(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  auto iters = c.equal_range(SearchedType<int>(1, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  iters = c.equal_range(SearchedType<int>(2, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  assert(conversions == 0);
  iters = c.equal_range(SearchedType<int>(3, &conversions));
  assert(std::distance(iters.first, iters.second) == 0);
  assert(conversions == 0);
}

template <typename Container, typename... Args>
void test_non_transparent_equal_range(Args&&... args) {
  Container c{std::forward<Args>(args)...};
  int conversions = 0;
  auto iters = c.equal_range(SearchedType<int>(1, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  assert(conversions > 0);
  conversions = 0;
  iters = c.equal_range(SearchedType<int>(2, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  assert(conversions > 0);
  conversions = 0;
  iters = c.equal_range(SearchedType<int>(3, &conversions));
  assert(std::distance(iters.first, iters.second) == 0);
  assert(conversions > 0);
}

#endif // TEST_STD_VER > 17

#endif // TEST_TRANSPARENT_UNORDERED_H
