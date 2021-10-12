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
#include <iostream>

// testing transparent unordered containers
#if TEST_STD_VER > 17

template <template <typename...> class UnorderedSet, typename Hash,
          typename Equal>
using unord_set_type = UnorderedSet<StoredType<int>, Hash, Equal>;

template <template <typename...> class UnorderedMap, typename Hash,
          typename Equal>
using unord_map_type = UnorderedMap<StoredType<int>, int, Hash, Equal>;

template <template <typename...> class UnorderedMap, typename Hash,
          typename Equal>
using unord_map_type2 = UnorderedMap<StoredType2<int>, int, Hash, Equal>;

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

template <typename Container>
void test_transparent_try_emplace() {
  Container c;

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

template <typename Container>
void test_non_transparent_try_emplace() {
  Container c;

  // Test successful copy insertion
  SearchedType<int> searched(1, nullptr);
  auto result1 = c.try_emplace(searched, 2);
  assert(result1.first != c.end());
  assert(result1.second);
  test_hetero_insertion_basic(result1.first, 1, 2, StoredCtorType::ST_Move, 2);

  // Test unsuccessful copy insertion
  auto result2 = c.try_emplace(searched, 3);
  assert(result2.first == result1.first);
  assert(!result2.second);
  test_hetero_insertion_basic(result2.first, 1, 2, StoredCtorType::ST_Move, 1);

  // Test successful insertion move
  auto result3 = c.try_emplace(SearchedType<int>(2, nullptr), 3);
  assert(result3.first != c.end());
  assert(result3.second);
  test_hetero_insertion_basic(result3.first, 2, 3, StoredCtorType::ST_Move, 2);

  // Test unsuccessful insertion move
  auto result4 = c.try_emplace(SearchedType<int>(2, nullptr), 4);
  assert(result4.first == result3.first);
  assert(!result4.second);
  test_hetero_insertion_basic(result4.first, 2, 3, StoredCtorType::ST_Move, 1);

  c.clear();
  // Test successful insertion copy hint
  auto result5 = c.try_emplace(c.begin(), searched, 2);
  assert(result5 != c.end());
  test_hetero_insertion_basic(result5, 1, 2, StoredCtorType::ST_Move, 2);

  // Test unsuccessful insertion copy hint
  auto result6 = c.try_emplace(c.begin(), searched, 3);
  assert(result6 == result5);
  test_hetero_insertion_basic(result6, 1, 2, StoredCtorType::ST_Move, 1);

  // Test successful insertion move hint
  auto result7 = c.try_emplace(c.begin(), SearchedType<int>(2, nullptr), 3);
  assert(result7 != c.end());
  test_hetero_insertion_basic(result7, 2, 3, StoredCtorType::ST_Move, 2);

  // Test unsuccessful insertion move hint
  auto result8 = c.try_emplace(c.begin(), SearchedType<int>(2, nullptr), 4);
  assert(result8 == result7);
  test_hetero_insertion_basic(result8, 2, 3, StoredCtorType::ST_Move, 1);
}

// enum class AssignTrackerType {
//   ST_NotAssigned = 0;
//   ST_LAssigned = 1,
//   ST_RAssigned = 2
// };

// struct AssignTracker {
//   AssignTracker(int) : assign_state(AssignTrackerType::ST_NotAssigned) {}

//   AssignTracker* operator=(const int&) {
//     assign_state = AssignTrackerType::ST_LAssigned;
//     return *this;
//   };

//   AssignTracker* operator=(int&&) {
//     assign_state = AssignTrackerType::ST_RAssigned;
//     return *this;
//   }

//   AssignTrackerType assign_state;
// };

// template <typename Container>
// void test_transparent_insert_or_assign() {
//   Container c;

//   // Test lvalue - expect insertion
//   SearchedType<int> searched(1, nullptr);
//   auto result1 = c.insert_or_assign(searched, 2);
//   assert(result1.first != c.end());
//   assert(result1.second);
//   assert(result1.first->second.assign_state == AssignTrackerType::ST_NotAssigned);
//   test_hetero_insertion_basic(result1.first, 1, 2, StoredCtorType::ST_LSearched, 1);

//   // Test lvalue - expect assignment
//   auto result2 = c.insert_or_assign(searched, 3);
//   assert(result2.first == result1.first);
//   assert(!result2.second);
//   assert(result1.first->second.assign_state == AssignTrackerType::ST_RAssigned);
//   test_hetero_insertion_basic(result2.first, 1, 3, StoredCtorType::ST_LSearched, 0);

//   // Test rvalue - expect insertion
//   int value = 1, value2 = 2;
//   auto result3 = c.insert_or_assign(SearchedType<int>(2, nullptr), value);
//   assert(result3.first != c.end());
//   assert(result3.second);
//   assert(result1.first->second.assign_state == AssignTrackerType::ST_NotAssigned);
//   test_hetero_insertion_basic(result3.first, 2, 1, StoredCtorType::ST_RSearched, 1);

//   // Test rvalue - expect assignment
//   auto result4 = c.insert_or_assign(SearchedType<int>(2, nullptr), value2);
//   assert(result4.first == result3.first);
//   assert(!result4.second);
//   assert(result1.first->second.assign_state == AssignTrackerType::ST_LAssigned);
//   test_hetero_insertion_basic(result4.first, 2, 2, StoredCtorType::ST_RSeached, 0);

//   c.clear();

//   // Test lvalue & hint - expect insertion
//   auto result5 = c.insert_or_assign(c.begin(), searched, 2);
//   assert(result5 != c.end());
//   test_hetero_insertion_basic(result5, 1, 2, StoredCtorType::ST_LSearched, 1);

//   // Test lvalue & hint - expect assignment
//   auto result6 = c.insert_or_assign(c.begin(), seached, 3);
//   assert(result6 == result5);
//   test_hetero_insertion_basic(result6, 1, 3, StoredCtorType::ST_LSearched, 0);

//   // Test rvalue & hint - expect insertion
//   auto result7 = c.insert_or_assign(c.begin(), SearchedType<int>(2, nullptr), 1);
//   assert(result7 != c.end());
//   test_hetero_insertion_basic(result7, 2, 1, StoredCtorType::ST_RSearched, 1);

//   // Test rvalue & hint - expect assignment
//   auto result8 = c.insert_or_assign(c.begin(), SearchedType<int>(2, nullptr), 2);
//   assert(result8 == result7);
//   test_hetero_insertion_basic(result8, 2, 2, StoredCtorType::ST_RSearched, 0);
// }

// template <typename Container>
// void test_non_transparent_insert_or_assign() {
//   Container c;

//   // Test lvalue - expect insertion
//   SearchedType<int> searched(1, nullptr);
//   auto result1 = c.insert_or_assign(searched, 2);
//   assert(result1.first != c.end());
//   assert(result1.second);
//   test_hetero_insertion_basic(result1.first, 1, 2, StoredCtorType::ST_Move, 2);

//   // Test lvalue - expect assignment
//   auto result2 = c.insert_or_assign(searched, 3);
//   assert(result2.first == result1.first);
//   assert(!result2.second);
//   test_hetero_insertion_basic(result2.first, 1, 3, StoredCtorType::ST_Move, 1);

//   // Test rvalue - expect insertion
//   auto result3 = c.insert_or_assign(SearchedType<int>(2, nullptr), 1);
//   assert(result3.first != c.end());
//   assert(result3.second);
//   test_hetero_insertion_basic(result3.first, 2, 1, StoredCtorType::ST_Move, 2);

//   // Test rvalue - expect assignment
//   auto result4 = c.insert_or_assign(SearchedType<int>(2, nullptr), 2);
//   assert(result4.first == result3.first);
//   assert(!result4.second);
//   test_hetero_insertion_basic(result4.first, 2, 2, StoredCtorType::ST_Move, 1);

//   c.clear();

//   // Test lvalue & hint - expect insertion
//   auto result5 = c.insert_or_assign(c.begin(), searched, 2);
//   assert(result5 != c.end());
//   test_hetero_insertion_basic(result5, 1, 2, StoredCtorType::ST_Move, 2);

//   // Test lvalue & hint - expect assignment
//   auto result6 = c.insert_or_assign(c.begin(), seached, 3);
//   assert(result6 == result5);
//   test_hetero_insertion_basic(result6, 1, 3, StoredCtorType::ST_Move, 1);

//   // Test rvalue & hint - expect insertion
//   auto result7 = c.insert_or_assign(c.begin(), SearchedType<int>(2, nullptr), 1);
//   assert(result7 != c.end());
//   test_hetero_insertion_basic(result7, 2, 1, StoredCtorType::ST_Move, 2);

//   // Test rvalue & hint - expect assignment
//   auto result8 = c.insert_or_assign(c.begin(), SearchedType<int>(2, nullptr), 2);
//   assert(result8 == result7);
//   test_hetero_insertion_basic(result8, 2, 2, StoredCtorType::ST_Move, 1);
// }

#endif // TEST_STD_VER > 17

#endif // TEST_TRANSPARENT_UNORDERED_H
