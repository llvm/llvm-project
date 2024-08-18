//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <vector>
#include <utility>

#include "benchmark/benchmark.h"

#include "ContainerBenchmarks.h"
#include "GenerateInput.h"
#include "test_macros.h"

using namespace ContainerBenchmarks;

constexpr size_t TestNumInputs = 1024;

template <class GenInputs>
inline auto getRandomWithMaxCardinality(GenInputs genInputs, size_t maxCardinality) {
  return [genInputs = std::forward<GenInputs>(genInputs), maxCardinality](size_t N) {
    auto possibleInputs = genInputs(maxCardinality);
    decltype(possibleInputs) inputs;
    size_t minIdx = 0, maxIdx = possibleInputs.size() - 1;
    for (size_t i = 0; i < N; ++i) {
      inputs.push_back(possibleInputs[getRandomInteger(minIdx, maxIdx)]);
    }
    return inputs;
  };
}

template <class GenInputsFirst, class GenInputsSecond>
inline auto genRandomPairInputs(GenInputsFirst genFirst, GenInputsSecond genSecond) {
  return [genFirst  = std::forward< GenInputsFirst >(genFirst),
          genSecond = std::forward< GenInputsSecond >(genSecond)](size_t N) {
    auto first  = genFirst(N);
    auto second = genSecond(N);
    std::vector<std::pair<typename decltype(first)::value_type, typename decltype(second)::value_type>> inputs;
    inputs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
      inputs.emplace_back(std::move(first[i]), std::move(second[i]));
    }
    return inputs;
  };
}

//----------------------------------------------------------------------------//
//                       BM_InsertValue
// ---------------------------------------------------------------------------//

BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getRandomIntegerInputs<uint32_t>, getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32_sorted,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getSortedIntegerInputs<uint32_t>, getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

// String //
BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_string,
                  std::unordered_multimap<std::string, uint32_t>{},
                  genRandomPairInputs(getRandomStringInputs, getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

// Prefixed String //
BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_prefixed_string,
                  std::unordered_multimap<std::string, uint32_t>{},
                  genRandomPairInputs(getPrefixedRandomStringInputs, getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

// Random with max cardinalities [512, 128, 32, 8, 1] //
BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32_max_cardinality_512,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 512),
                                      getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32_max_cardinality_128,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 128),
                                      getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32_max_cardinality_32,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 32),
                                      getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32_max_cardinality_8,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 8),
                                      getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_InsertValue,
                  unordered_multimap_uint32_max_cardinality_1,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  genRandomPairInputs(getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 1),
                                      getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertValue,
    unordered_multimap_string_max_cardinality_512,
    std::unordered_multimap<std::string, uint32_t>{},
    genRandomPairInputs(getRandomWithMaxCardinality(getRandomStringInputs, 512), getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertValue,
    unordered_multimap_string_max_cardinality_128,
    std::unordered_multimap<std::string, uint32_t>{},
    genRandomPairInputs(getRandomWithMaxCardinality(getRandomStringInputs, 128), getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertValue,
    unordered_multimap_string_max_cardinality_32,
    std::unordered_multimap<std::string, uint32_t>{},
    genRandomPairInputs(getRandomWithMaxCardinality(getRandomStringInputs, 32), getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertValue,
    unordered_multimap_string_max_cardinality_8,
    std::unordered_multimap<std::string, uint32_t>{},
    genRandomPairInputs(getRandomWithMaxCardinality(getRandomStringInputs, 8), getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(
    BM_InsertValue,
    unordered_multimap_string_max_cardinality_1,
    std::unordered_multimap<std::string, uint32_t>{},
    genRandomPairInputs(getRandomWithMaxCardinality(getRandomStringInputs, 1), getRandomIntegerInputs<uint32_t>))
    ->Arg(TestNumInputs);

//----------------------------------------------------------------------------//
//                       BM_EmplaceValue
// ---------------------------------------------------------------------------//

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getRandomIntegerInputs<uint32_t>,
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32_sorted,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getSortedIntegerInputs<uint32_t>,
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

// String //
BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_string,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getRandomStringInputs,
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

// Prefixed String //
BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_prefixed_string,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getPrefixedRandomStringInputs,
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

// Random with max cardinalities [512, 128, 32, 8] //
BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32_max_cardinality_512,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 512),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32_max_cardinality_128,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 128),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32_max_cardinality_32,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 32),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32_max_cardinality_8,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 8),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_uint32_max_cardinality_1,
                  std::unordered_multimap<uint32_t, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomIntegerInputs<uint32_t>, 1),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_string_max_cardinality_512,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomStringInputs, 512),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_string_max_cardinality_128,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomStringInputs, 128),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_string_max_cardinality_32,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomStringInputs, 32),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_string_max_cardinality_8,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomStringInputs, 8),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_EmplaceValue,
                  unordered_multimap_string_max_cardinality_1,
                  std::unordered_multimap<std::string, uint32_t>{},
                  getRandomWithMaxCardinality(getRandomStringInputs, 1),
                  getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_MAIN();
