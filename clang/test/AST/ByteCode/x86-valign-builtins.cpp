// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512dq -verify=expected -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512dq -verify=ref %s

// expected-no-diagnostics
// ref-no-diagnostics

#define __MM_MALLOC_H
#include <immintrin.h>

using v4si = int __attribute__((vector_size(16)));
using v8si = int __attribute__((vector_size(32)));
using v16si = int __attribute__((vector_size(64)));
using v4di = long long __attribute__((vector_size(32)));

constexpr v4si test_alignr_epi32_128() {
  v4si A = {100, 200, 300, 400};
  v4si B = {10, 20, 30, 40};
  return (v4si)_mm_alignr_epi32((__m128i)A, (__m128i)B, 1);
}

constexpr v8si test_alignr_epi32_256() {
  v8si A = {100, 200, 300, 400, 500, 600, 700, 800};
  v8si B = {1, 2, 3, 4, 5, 6, 7, 8};
  return (v8si)_mm256_alignr_epi32((__m256i)A, (__m256i)B, 3);
}

constexpr v16si test_alignr_epi32_512_wrap() {
  v16si A = {100, 200, 300, 400, 500, 600, 700, 800,
             900, 1000, 1100, 1200, 1300, 1400, 1500, 1600};
  v16si B = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return (v16si)_mm512_alignr_epi32((__m512i)A, (__m512i)B, 19);
}

constexpr v4di test_alignr_epi64_256() {
  v4di A = {10, 11, 12, 13};
  v4di B = {1, 2, 3, 4};
  return (v4di)_mm256_alignr_epi64((__m256i)A, (__m256i)B, 2);
}

constexpr v4si R128 = test_alignr_epi32_128();
static_assert(R128[0] == 20 && R128[1] == 30 && R128[2] == 40 && R128[3] == 100);

constexpr v8si R256 = test_alignr_epi32_256();
static_assert(R256[0] == 4 && R256[1] == 5 && R256[2] == 6 && R256[3] == 7);
static_assert(R256[4] == 8 && R256[5] == 100 && R256[6] == 200 && R256[7] == 300);

constexpr v16si R512 = test_alignr_epi32_512_wrap();
static_assert(R512[0] == 3 && R512[1] == 4 && R512[2] == 5 && R512[3] == 6);
static_assert(R512[8] == 11 && R512[9] == 12 && R512[10] == 13 && R512[11] == 14);
static_assert(R512[12] == 15 && R512[13] == 100 && R512[14] == 200 && R512[15] == 300);

constexpr v4di R64 = test_alignr_epi64_256();
static_assert(R64[0] == 3 && R64[1] == 4 && R64[2] == 10 && R64[3] == 11);
