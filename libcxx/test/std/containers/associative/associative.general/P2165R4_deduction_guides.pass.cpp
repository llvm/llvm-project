//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <array>
#include <flat_map>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "test_macros.h"

int main(int, char**) {
  // --- Input Data ---
  // 1. Vector of std::pair
  std::vector<std::pair<const int, std::string>> pair_vec = {{1, "apple"}, {2, "banana"}, {3, "cherry"}};

  // 2. Vector of std::tuple
  std::vector<std::tuple<int, double>> tuple_vec = {{10, 1.1}, {20, 2.2}, {30, 3.3}};

  // 3. Vector of std::array
  std::vector<std::array<long, 2>> array_vec = {{100L, 101L}, {200L, 201L}, {300L, 301L}};

  // 4. Vector of std::pair with non-const key (for testing const addition in iter_to_alloc_type)
  std::vector<std::pair<int, std::string>> non_const_key_pair_vec = {{5, "grape"}, {6, "kiwi"}};

  // --- CTAD Tests ---

  // map
  std::map m1(pair_vec.begin(), pair_vec.end());
  static_assert(std::is_same_v<decltype(m1), std::map<int, std::string>>);

  // multimap
  std::multimap mm1(pair_vec.begin(), pair_vec.end());
  static_assert(std::is_same_v<decltype(mm1), std::multimap<int, std::string>>);

  // unordered_map
  std::unordered_map um1(pair_vec.begin(), pair_vec.end());
  static_assert(std::is_same_v<decltype(um1), std::unordered_map<int, std::string>>);

  // unordered_multimap
  std::unordered_multimap umm1(pair_vec.begin(), pair_vec.end());
  static_assert(std::is_same_v<decltype(umm1), std::unordered_multimap<int, std::string>>);

  // flat_map
  std::flat_map fm1(pair_vec.begin(), pair_vec.end());
  static_assert(std::is_same_v<decltype(fm1), std::flat_map<int, std::string>>);

  // flat_multimap
  std::flat_multimap fmm1(pair_vec.begin(), pair_vec.end());
  static_assert(std::is_same_v<decltype(fmm1), std::flat_multimap<int, std::string>>);

  // map
  std::map m2(tuple_vec.begin(), tuple_vec.end());
  static_assert(std::is_same_v<decltype(m2), std::map<int, double>>);

  // multimap
  std::multimap mm2(tuple_vec.begin(), tuple_vec.end());
  static_assert(std::is_same_v<decltype(mm2), std::multimap<int, double>>);

  // unordered_map
  std::unordered_map um2(tuple_vec.begin(), tuple_vec.end());
  // Note: std::tuple needs a hash specialization to be used as a key in unordered containers.
  // CTAD itself should work, but compilation/runtime might fail without a hash.
  // This static_assert checks the deduced type. A hash specialization would be needed for actual use.
  static_assert(std::is_same_v<decltype(um2), std::unordered_map<int, double>>);

  // unordered_multimap
  std::unordered_multimap umm2(tuple_vec.begin(), tuple_vec.end());
  static_assert(std::is_same_v<decltype(umm2), std::unordered_multimap<int, double>>);

  // flat_map
  std::flat_map fm2(tuple_vec.begin(), tuple_vec.end());
  static_assert(std::is_same_v<decltype(fm2), std::flat_map<int, double>>);

  // flat_multimap
  std::flat_multimap fmm2(tuple_vec.begin(), tuple_vec.end());
  static_assert(std::is_same_v<decltype(fmm2), std::flat_multimap<int, double>>);

  // map
  std::map m3(array_vec.begin(), array_vec.end());
  static_assert(std::is_same_v<decltype(m3), std::map<long, long>>);

  // multimap
  std::multimap mm3(array_vec.begin(), array_vec.end());
  static_assert(std::is_same_v<decltype(mm3), std::multimap<long, long>>);

  // unordered_map
  std::unordered_map um3(array_vec.begin(), array_vec.end());
  // Note: std::array needs a hash specialization.
  static_assert(std::is_same_v<decltype(um3), std::unordered_map<long, long>>);

  // unordered_multimap
  std::unordered_multimap umm3(array_vec.begin(), array_vec.end());
  static_assert(std::is_same_v<decltype(umm3), std::unordered_multimap<long, long>>);

  // flat_map
  std::flat_map fm3(array_vec.begin(), array_vec.end());
  static_assert(std::is_same_v<decltype(fm3), std::flat_map<long, long>>);

  // flat_multimap
  std::flat_multimap fmm3(array_vec.begin(), array_vec.end());
  static_assert(std::is_same_v<decltype(fmm3), std::flat_multimap<long, long>>);

  // map
  std::map m4(non_const_key_pair_vec.begin(), non_const_key_pair_vec.end());
  static_assert(std::is_same_v<decltype(m4), std::map<int, std::string>>);

  return 0;
}
