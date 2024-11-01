//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cstdint>
#include <cstddef>
#include <functional>

#include "benchmark/benchmark.h"

#include "GenerateInput.h"
#include "test_macros.h"

constexpr std::size_t TestNumInputs = 1024;

template <class _Size>
inline TEST_ALWAYS_INLINE _Size loadword(const void* __p) {
  _Size __r;
  std::memcpy(&__r, __p, sizeof(__r));
  return __r;
}

inline TEST_ALWAYS_INLINE std::size_t hash_len_16(std::size_t __u, std::size_t __v) {
  const std::size_t __mul = 0x9ddfea08eb382d69ULL;
  std::size_t __a         = (__u ^ __v) * __mul;
  __a ^= (__a >> 47);
  std::size_t __b = (__v ^ __a) * __mul;
  __b ^= (__b >> 47);
  __b *= __mul;
  return __b;
}

template <std::size_t _Len>
inline TEST_ALWAYS_INLINE std::size_t hash_len_0_to_8(const char* __s) {
  static_assert(_Len == 4 || _Len == 8, "");
  const uint64_t __a = loadword<uint32_t>(__s);
  const uint64_t __b = loadword<uint32_t>(__s + _Len - 4);
  return hash_len_16(_Len + (__a << 3), __b);
}

struct UInt32Hash {
  UInt32Hash() = default;
  inline TEST_ALWAYS_INLINE std::size_t operator()(uint32_t data) const {
    return hash_len_0_to_8<4>(reinterpret_cast<const char*>(&data));
  }
};

template <class HashFn, class GenInputs>
void BM_Hash(benchmark::State& st, HashFn fn, GenInputs gen) {
  auto in               = gen(st.range(0));
  const auto end        = in.data() + in.size();
  std::size_t last_hash = 0;
  benchmark::DoNotOptimize(&last_hash);
  while (st.KeepRunning()) {
    for (auto it = in.data(); it != end; ++it) {
      benchmark::DoNotOptimize(last_hash += fn(*it));
    }
    benchmark::ClobberMemory();
  }
}

BENCHMARK_CAPTURE(BM_Hash, uint32_random_std_hash, std::hash<uint32_t>{}, getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Hash, uint32_random_custom_hash, UInt32Hash{}, getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Hash, uint32_top_std_hash, std::hash<uint32_t>{}, getSortedTopBitsIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Hash, uint32_top_custom_hash, UInt32Hash{}, getSortedTopBitsIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_MAIN();
