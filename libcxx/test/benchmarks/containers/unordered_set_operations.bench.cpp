//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_set>
#include <vector>

#include "benchmark/benchmark.h"

#include "ContainerBenchmarks.h"
#include "../GenerateInput.h"
#include "test_macros.h"

using namespace ContainerBenchmarks;

constexpr std::size_t TestNumInputs = 1024;

// The purpose of this hash function is to NOT be implemented as the identity function,
// which is how std::hash is implemented for smaller integral types.
struct NonIdentityScalarHash : std::hash<unsigned long long> {};

// The sole purpose of this comparator is to be used in BM_Rehash, where
// we need something slow enough to be easily noticable in benchmark results.
// The default implementation of operator== for strings seems to be a little
// too fast for that specific benchmark to reliably show a noticeable
// improvement, but unoptimized bytewise comparison fits just right.
// Early return is there just for convenience, since we only compare strings
// of equal length in BM_Rehash.
struct SlowStringEq {
  SlowStringEq() = default;
  inline TEST_ALWAYS_INLINE bool operator()(const std::string& lhs, const std::string& rhs) const {
    if (lhs.size() != rhs.size())
      return false;

    bool eq = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
      eq &= lhs[i] == rhs[i];
    }
    return eq;
  }
};

//----------------------------------------------------------------------------//
//                       BM_InsertValue
// ---------------------------------------------------------------------------//

// Sorted Ascending //
BENCHMARK_CAPTURE(
    BM_InsertValue, unordered_set_uint32, std::unordered_set<uint32_t>{}, getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertValue, unordered_set_uint32_sorted, std::unordered_set<uint32_t>{}, getSortedIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

// Top Bytes //
BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_set_top_bits_uint32,
                  std::unordered_set<uint32_t>{},
                  getSortedTopBitsIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValueRehash,
                  unordered_set_top_bits_uint32,
                  std::unordered_set<uint32_t, NonIdentityScalarHash>{},
                  getSortedTopBitsIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

// String //
BENCHMARK_CAPTURE(BM_InsertValue, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValueRehash, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

// Prefixed String //
BENCHMARK_CAPTURE(
    BM_InsertValue, unordered_set_prefixed_string, std::unordered_set<std::string>{}, getPrefixedRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValueRehash,
                  unordered_set_prefixed_string,
                  std::unordered_set<std::string>{},
                  getPrefixedRandomStringInputs)
    ->Arg(TestNumInputs);

//----------------------------------------------------------------------------//
//                         BM_Find
// ---------------------------------------------------------------------------//

// Random //
BENCHMARK_CAPTURE(
    BM_Find, unordered_set_random_uint64, std::unordered_set<uint64_t>{}, getRandomIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash,
                  unordered_set_random_uint64,
                  std::unordered_set<uint64_t, NonIdentityScalarHash>{},
                  getRandomIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

// Sorted //
BENCHMARK_CAPTURE(
    BM_Find, unordered_set_sorted_uint64, std::unordered_set<uint64_t>{}, getSortedIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash,
                  unordered_set_sorted_uint64,
                  std::unordered_set<uint64_t, NonIdentityScalarHash>{},
                  getSortedIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

// Sorted //
#ifndef TEST_HAS_NO_INT128
BENCHMARK_CAPTURE(BM_Find,
                  unordered_set_sorted_uint128,
                  std::unordered_set<__uint128_t>{},
                  getSortedTopBitsIntegerInputs<__uint128_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash,
                  unordered_set_sorted_uint128,
                  std::unordered_set<__uint128_t>{},
                  getSortedTopBitsIntegerInputs<__uint128_t>)
    ->Arg(TestNumInputs);
#endif

// Sorted //
BENCHMARK_CAPTURE(
    BM_Find, unordered_set_sorted_uint32, std::unordered_set<uint32_t>{}, getSortedIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash,
                  unordered_set_sorted_uint32,
                  std::unordered_set<uint32_t, NonIdentityScalarHash>{},
                  getSortedIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

// Sorted Ascending //
BENCHMARK_CAPTURE(
    BM_Find, unordered_set_sorted_large_uint64, std::unordered_set<uint64_t>{}, getSortedLargeIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash,
                  unordered_set_sorted_large_uint64,
                  std::unordered_set<uint64_t, NonIdentityScalarHash>{},
                  getSortedLargeIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

// Top Bits //
BENCHMARK_CAPTURE(
    BM_Find, unordered_set_top_bits_uint64, std::unordered_set<uint64_t>{}, getSortedTopBitsIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash,
                  unordered_set_top_bits_uint64,
                  std::unordered_set<uint64_t, NonIdentityScalarHash>{},
                  getSortedTopBitsIntegerInputs<uint64_t>)
    ->Arg(TestNumInputs);

// String //
BENCHMARK_CAPTURE(BM_Find, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_FindRehash, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

// Prefixed String //
BENCHMARK_CAPTURE(
    BM_Find, unordered_set_prefixed_string, std::unordered_set<std::string>{}, getPrefixedRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_FindRehash, unordered_set_prefixed_string, std::unordered_set<std::string>{}, getPrefixedRandomStringInputs)
    ->Arg(TestNumInputs);

//----------------------------------------------------------------------------//
//                         BM_Rehash
// ---------------------------------------------------------------------------//

BENCHMARK_CAPTURE(BM_Rehash,
                  unordered_set_string_arg,
                  std::unordered_set<std::string, std::hash<std::string>, SlowStringEq>{},
                  getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Rehash, unordered_set_int_arg, std::unordered_set<int>{}, getRandomIntegerInputs<int>)
    ->Arg(TestNumInputs);

//----------------------------------------------------------------------------//
//                         BM_Compare
// ---------------------------------------------------------------------------//

BENCHMARK_CAPTURE(
    BM_Compare_same_container, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Compare_same_container, unordered_set_int, std::unordered_set<int>{}, getRandomIntegerInputs<int>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_Compare_different_containers, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_Compare_different_containers, unordered_set_int, std::unordered_set<int>{}, getRandomIntegerInputs<int>)
    ->Arg(TestNumInputs);

///////////////////////////////////////////////////////////////////////////////
BENCHMARK_CAPTURE(BM_InsertDuplicate, unordered_set_int, std::unordered_set<int>{}, getRandomIntegerInputs<int>)
    ->Arg(TestNumInputs);
BENCHMARK_CAPTURE(BM_InsertDuplicate, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceDuplicate, unordered_set_int, std::unordered_set<int>{}, getRandomIntegerInputs<int>)
    ->Arg(TestNumInputs);
BENCHMARK_CAPTURE(BM_EmplaceDuplicate, unordered_set_string, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertDuplicate, unordered_set_int_insert_arg, std::unordered_set<int>{}, getRandomIntegerInputs<int>)
    ->Arg(TestNumInputs);
BENCHMARK_CAPTURE(
    BM_InsertDuplicate, unordered_set_string_insert_arg, std::unordered_set<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_EmplaceDuplicate, unordered_set_int_insert_arg, std::unordered_set<int>{}, getRandomIntegerInputs<unsigned>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_EmplaceDuplicate, unordered_set_string_arg, std::unordered_set<std::string>{}, getRandomCStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_MAIN();
