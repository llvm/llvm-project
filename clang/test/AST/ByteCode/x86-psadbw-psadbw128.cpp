// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O0 \
// RUN:   -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O0 \
// RUN:   -target-feature +avx2 -target-feature +avx512bw \
// RUN:   -fexperimental-new-constant-interpreter -verify %s

// No headers allowed. Use Clang's vector types directly.
// expected-no-diagnostics

typedef char v16qi __attribute__((vector_size(16)));
typedef char v32qi __attribute__((vector_size(32)));
typedef char v64qi __attribute__((vector_size(64)));
typedef long long v2di __attribute__((vector_size(16)));
typedef long long v4di __attribute__((vector_size(32)));
typedef long long v8di __attribute__((vector_size(64)));

constexpr v2di test_psadbw128() {
  v16qi a = {10,20,30,40,50,60,70,80,
             1,2,3,4,5,6,7,8};

  v16qi b = {5,15,25,45,55,55,75,85,
             10,0,3,9,1,10,2,8};

  // Call the builtin directly
  return __builtin_ia32_psadbw128(a, b);
}

static_assert(test_psadbw128()[0] == 40, "block0 mismatch");
static_assert(test_psadbw128()[1] == 29, "block1 mismatch");

#ifdef __AVX2__
constexpr v4di test_psadbw256() {
  v32qi a = {
      0, 1, 2, 3, 4, 5, 6, 7,     8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  v32qi b = {};
  return __builtin_ia32_psadbw256(a, b);
}

static_assert(test_psadbw256()[0] == 28, "avx2 block0 mismatch");
static_assert(test_psadbw256()[1] == 92, "avx2 block1 mismatch");
static_assert(test_psadbw256()[2] == 156, "avx2 block2 mismatch");
static_assert(test_psadbw256()[3] == 220, "avx2 block3 mismatch");
#endif

#ifdef __AVX512BW__
constexpr v8di test_psadbw512() {
  v64qi a = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  v64qi b = {};
  return __builtin_ia32_psadbw512(a, b);
}

static_assert(test_psadbw512()[0] == 28, "avx512 block0 mismatch");
static_assert(test_psadbw512()[1] == 92, "avx512 block1 mismatch");
static_assert(test_psadbw512()[2] == 156, "avx512 block2 mismatch");
static_assert(test_psadbw512()[3] == 220, "avx512 block3 mismatch");
static_assert(test_psadbw512()[4] == 284, "avx512 block4 mismatch");
static_assert(test_psadbw512()[5] == 348, "avx512 block5 mismatch");
static_assert(test_psadbw512()[6] == 412, "avx512 block6 mismatch");
static_assert(test_psadbw512()[7] == 476, "avx512 block7 mismatch");
#endif
