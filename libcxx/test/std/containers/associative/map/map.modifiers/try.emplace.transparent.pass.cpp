//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

// <map>

// class map

// template <class K, class... Args>
//  pair<iterator, bool> try_emplace(K&& k, Args&&... args);               // C++XX
// template <class K, class... Args>
//  pair<iterator, bool> try_emplace(K&& k, Args&&... args);               // C++XX
// template <class K, class... Args>
//  iterator try_emplace(const_iterator hint, K&& k, Args&&... args);      // C++XX
// template <class K, class... Args>
//  iterator try_emplace(const_iterator hint, K&& k, Args&&... args);      // C++XX

#include <map>
#include <cassert>

#include "is_transparent.h"

using stored_type = StoredType2<int>;
using searched_type = SearchedType<int>;
using map_type = std::map<stored_type, int, std::less<>>;

template <typename It, typename T, typename U>
void test_hetero_insertion_basic(It actual_result, const T& expected_key,
                                 const U& expected_value, StoredCtorType expected_construction,
                                 int expected_n_constructions)
{
  assert(actual_result->first.get_value() == expected_key);
  assert(actual_result->second == expected_value);
  assert(actual_result->first.construction == expected_construction);
  assert(StoredType2<T>::n_constructions == expected_n_constructions);
  StoredType2<T>::reset();
}

void test_transparent() {
    map_type c;

    // Test successful copy insertion
  SearchedType<int> searched(1, nullptr);
  auto result1 = c.try_emplace(searched, 2);
  assert(result1.first != c.end());
  assert(result1.second);
  test_hetero_insertion_basic(result1.first, 1, 2, StoredCtorType::ST_LSearched, 1);

  // Test unsuccessful copy insertion
  auto result2 = c.try_emplace(searched, 3);
  assert(result2.first == result1.first);
  assert(!result2.second);
  test_hetero_insertion_basic(result2.first, 1, 2, StoredCtorType::ST_LSearched, 0);

  // Test successful insertion move
  auto result3 = c.try_emplace(SearchedType<int>(2, nullptr), 3);
  assert(result3.first != c.end());
  assert(result3.second);
  test_hetero_insertion_basic(result3.first, 2, 3, StoredCtorType::ST_RSearched, 1);

  // Test unsuccessful insertion move
  auto result4 = c.try_emplace(SearchedType<int>(2, nullptr), 4);
  assert(result4.first == result3.first);
  assert(!result4.second);
  test_hetero_insertion_basic(result4.first, 2, 3, StoredCtorType::ST_RSearched, 0);

  c.clear();
  // Test successful insertion copy hint
  auto result5 = c.try_emplace(c.begin(), searched, 2);
  assert(result5 != c.end());
  test_hetero_insertion_basic(result5, 1, 2, StoredCtorType::ST_LSearched, 1);

  // Test unsuccessful insertion copy hint
  auto result6 = c.try_emplace(c.begin(), searched, 3);
  assert(result6 == result5);
  test_hetero_insertion_basic(result6, 1, 2, StoredCtorType::ST_LSearched, 0);

  // Test successful insertion move hint
  auto result7 = c.try_emplace(c.begin(), SearchedType<int>(2, nullptr), 3);
  assert(result7 != c.end());
  test_hetero_insertion_basic(result7, 2, 3, StoredCtorType::ST_RSearched, 1);

  // Test unsuccessful insertion move hint
  auto result8 = c.try_emplace(c.begin(), SearchedType<int>(2, nullptr), 4);
  assert(result8 == result7);
  test_hetero_insertion_basic(result8, 2, 3, StoredCtorType::ST_RSearched, 0);
}

int main() {
    test_transparent();
}