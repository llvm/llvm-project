// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_atomic
//===-- atomic_test.c - Test support functions for atomic operations ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file performs some simple testing of the support functions for the
// atomic builtins. All tests are single-threaded, so this is only a sanity
// check.
//
//===----------------------------------------------------------------------===//

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// We directly test the library atomic functions, not using the C builtins. This
// should avoid confounding factors, ensuring that we actually test the
// functions themselves, regardless of how the builtins are lowered. We need to
// use asm labels because we can't redeclare the builtins.
// Note: we need to prepend an underscore to this name for e.g. macOS.
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define EXTERNAL_NAME(name) asm(STRINGIFY(__USER_LABEL_PREFIX__) #name)

void __atomic_load_c(int size, void *src, void *dest,
                     int model) EXTERNAL_NAME(__atomic_load);

uint8_t __atomic_load_1(uint8_t *src, int model);
uint16_t __atomic_load_2(uint16_t *src, int model);
uint32_t __atomic_load_4(uint32_t *src, int model);
uint64_t __atomic_load_8(uint64_t *src, int model);

void __atomic_store_c(int size, void *dest, const void *src,
                      int model) EXTERNAL_NAME(__atomic_store);

void __atomic_store_1(uint8_t *dest, uint8_t val, int model);
void __atomic_store_2(uint16_t *dest, uint16_t val, int model);
void __atomic_store_4(uint32_t *dest, uint32_t val, int model);
void __atomic_store_8(uint64_t *dest, uint64_t val, int model);

void __atomic_exchange_c(int size, void *ptr, const void *val, void *old,
                         int model) EXTERNAL_NAME(__atomic_exchange);

uint8_t __atomic_exchange_1(uint8_t *dest, uint8_t val, int model);
uint16_t __atomic_exchange_2(uint16_t *dest, uint16_t val, int model);
uint32_t __atomic_exchange_4(uint32_t *dest, uint32_t val, int model);
uint64_t __atomic_exchange_8(uint64_t *dest, uint64_t val, int model);

int __atomic_compare_exchange_c(int size, void *ptr, void *expected,
                                const void *desired, int success, int failure)
    EXTERNAL_NAME(__atomic_compare_exchange);

bool __atomic_compare_exchange_1(uint8_t *ptr, uint8_t *expected,
                                 uint8_t desired, int success, int failure);
bool __atomic_compare_exchange_2(uint16_t *ptr, uint16_t *expected,
                                 uint16_t desired, int success, int failure);
bool __atomic_compare_exchange_4(uint32_t *ptr, uint32_t *expected,
                                 uint32_t desired, int success, int failure);
bool __atomic_compare_exchange_8(uint64_t *ptr, uint64_t *expected,
                                 uint64_t desired, int success, int failure);

uint8_t __atomic_fetch_add_1(uint8_t *ptr, uint8_t val, int model);
uint16_t __atomic_fetch_add_2(uint16_t *ptr, uint16_t val, int model);
uint32_t __atomic_fetch_add_4(uint32_t *ptr, uint32_t val, int model);
uint64_t __atomic_fetch_add_8(uint64_t *ptr, uint64_t val, int model);

uint8_t __atomic_fetch_sub_1(uint8_t *ptr, uint8_t val, int model);
uint16_t __atomic_fetch_sub_2(uint16_t *ptr, uint16_t val, int model);
uint32_t __atomic_fetch_sub_4(uint32_t *ptr, uint32_t val, int model);
uint64_t __atomic_fetch_sub_8(uint64_t *ptr, uint64_t val, int model);

uint8_t __atomic_fetch_and_1(uint8_t *ptr, uint8_t val, int model);
uint16_t __atomic_fetch_and_2(uint16_t *ptr, uint16_t val, int model);
uint32_t __atomic_fetch_and_4(uint32_t *ptr, uint32_t val, int model);
uint64_t __atomic_fetch_and_8(uint64_t *ptr, uint64_t val, int model);

uint8_t __atomic_fetch_or_1(uint8_t *ptr, uint8_t val, int model);
uint16_t __atomic_fetch_or_2(uint16_t *ptr, uint16_t val, int model);
uint32_t __atomic_fetch_or_4(uint32_t *ptr, uint32_t val, int model);
uint64_t __atomic_fetch_or_8(uint64_t *ptr, uint64_t val, int model);

uint8_t __atomic_fetch_xor_1(uint8_t *ptr, uint8_t val, int model);
uint16_t __atomic_fetch_xor_2(uint16_t *ptr, uint16_t val, int model);
uint32_t __atomic_fetch_xor_4(uint32_t *ptr, uint32_t val, int model);
uint64_t __atomic_fetch_xor_8(uint64_t *ptr, uint64_t val, int model);

// We conditionally test the *_16 atomic function variants based on the same
// condition that compiler_rt (atomic.c) uses to conditionally generate them.
// Currently atomic.c tests if __SIZEOF_INT128__ is defined (which can be the
// case on 32-bit platforms, by using -fforce-enable-int128), instead of using
// CRT_HAS_128BIT.

#ifdef __SIZEOF_INT128__
#define TEST_16
#endif

#ifdef TEST_16
typedef __uint128_t uint128_t;
typedef uint128_t maxuint_t;
uint128_t __atomic_load_16(uint128_t *src, int model);
void __atomic_store_16(uint128_t *dest, uint128_t val, int model);
uint128_t __atomic_exchange_16(uint128_t *dest, uint128_t val, int model);
bool __atomic_compare_exchange_16(uint128_t *ptr, uint128_t *expected,
                                  uint128_t desired, int success, int failure);
uint128_t __atomic_fetch_add_16(uint128_t *ptr, uint128_t val, int model);
uint128_t __atomic_fetch_sub_16(uint128_t *ptr, uint128_t val, int model);
uint128_t __atomic_fetch_and_16(uint128_t *ptr, uint128_t val, int model);
uint128_t __atomic_fetch_or_16(uint128_t *ptr, uint128_t val, int model);
uint128_t __atomic_fetch_xor_16(uint128_t *ptr, uint128_t val, int model);
#else
typedef uint64_t maxuint_t;
#endif

#define U8(value) ((uint8_t)(value))
#define U16(value) ((uint16_t)(value))
#define U32(value) ((uint32_t)(value))
#define U64(value) ((uint64_t)(value))

#ifdef TEST_16
#define V ((((uint128_t)0x4243444546474849) << 64) | 0x4a4b4c4d4e4f5051)
#define ONES ((((uint128_t)0x0101010101010101) << 64) | 0x0101010101010101)
#else
#define V 0x4243444546474849
#define ONES 0x0101010101010101
#endif

#define LEN(array) (sizeof(array) / sizeof(array[0]))

__attribute__((aligned(16))) static const char data[] = {
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
};

uint8_t a8, b8;
uint16_t a16, b16;
uint32_t a32, b32;
uint64_t a64, b64;
#ifdef TEST_16
uint128_t a128, b128;
#endif

void set_a_values(maxuint_t value) {
  a8 = U8(value);
  a16 = U16(value);
  a32 = U32(value);
  a64 = U64(value);
#ifdef TEST_16
  a128 = value;
#endif
}

void set_b_values(maxuint_t value) {
  b8 = U8(value);
  b16 = U16(value);
  b32 = U32(value);
  b64 = U64(value);
#ifdef TEST_16
  b128 = value;
#endif
}

void test_loads(void) {
  static int atomic_load_models[] = {
      __ATOMIC_RELAXED,
      __ATOMIC_CONSUME,
      __ATOMIC_ACQUIRE,
      __ATOMIC_SEQ_CST,
  };

  for (int m = 0; m < LEN(atomic_load_models); m++) {
    int model = atomic_load_models[m];

    // Test with aligned data.
    for (int n = 1; n <= LEN(data); n++) {
      __attribute__((aligned(16))) char dst[LEN(data)] = {0};
      __atomic_load_c(n, data, dst, model);
      if (memcmp(dst, data, n) != 0)
        abort();
    }

    // Test with unaligned data.
    for (int n = 1; n < LEN(data); n++) {
      __attribute__((aligned(16))) char dst[LEN(data)] = {0};
      __atomic_load_c(n, data + 1, dst + 1, model);
      if (memcmp(dst + 1, data + 1, n) != 0)
        abort();
    }

    set_a_values(V + m);
    if (__atomic_load_1(&a8, model) != U8(V + m))
      abort();
    if (__atomic_load_2(&a16, model) != U16(V + m))
      abort();
    if (__atomic_load_4(&a32, model) != U32(V + m))
      abort();
    if (__atomic_load_8(&a64, model) != U64(V + m))
      abort();
#ifdef TEST_16
    if (__atomic_load_16(&a128, model) != V + m)
      abort();
#endif
  }
}

void test_stores(void) {
  static int atomic_store_models[] = {
      __ATOMIC_RELAXED,
      __ATOMIC_RELEASE,
      __ATOMIC_SEQ_CST,
  };

  for (int m = 0; m < LEN(atomic_store_models); m++) {
    int model = atomic_store_models[m];

    // Test with aligned data.
    for (int n = 1; n <= LEN(data); n++) {
      __attribute__((aligned(16))) char dst[LEN(data)];
      __atomic_store_c(n, dst, data, model);
      if (memcmp(data, dst, n) != 0)
        abort();
    }

    // Test with unaligned data.
    for (int n = 1; n < LEN(data); n++) {
      __attribute__((aligned(16))) char dst[LEN(data)];
      __atomic_store_c(n, dst + 1, data + 1, model);
      if (memcmp(data + 1, dst + 1, n) != 0)
        abort();
    }

    __atomic_store_1(&a8, U8(V + m), model);
    if (a8 != U8(V + m))
      abort();
    __atomic_store_2(&a16, U16(V + m), model);
    if (a16 != U16(V + m))
      abort();
    __atomic_store_4(&a32, U32(V + m), model);
    if (a32 != U32(V + m))
      abort();
    __atomic_store_8(&a64, U64(V + m), model);
    if (a64 != U64(V + m))
      abort();
#ifdef TEST_16
    __atomic_store_16(&a128, V + m, model);
    if (a128 != V + m)
      abort();
#endif
  }
}

void test_exchanges(void) {
  static int atomic_exchange_models[] = {
      __ATOMIC_RELAXED,
      __ATOMIC_ACQUIRE,
      __ATOMIC_RELEASE,
      __ATOMIC_ACQ_REL,
      __ATOMIC_SEQ_CST,
  };

  set_a_values(V);

  for (int m = 0; m < LEN(atomic_exchange_models); m++) {
    int model = atomic_exchange_models[m];

    // Test with aligned data.
    for (int n = 1; n <= LEN(data); n++) {
      __attribute__((aligned(16))) char dst[LEN(data)];
      __attribute__((aligned(16))) char old[LEN(data)];
      for (int i = 0; i < LEN(dst); i++)
        dst[i] = i + m;
      __atomic_exchange_c(n, dst, data, old, model);
      for (int i = 0; i < n; i++) {
        if (dst[i] != 0x10 + i || old[i] != i + m)
          abort();
      }
    }

    // Test with unaligned data.
    for (int n = 1; n < LEN(data); n++) {
      __attribute__((aligned(16))) char dst[LEN(data)];
      __attribute__((aligned(16))) char old[LEN(data)];
      for (int i = 1; i < LEN(dst); i++)
        dst[i] = i - 1 + m;
      __atomic_exchange_c(n, dst + 1, data + 1, old + 1, model);
      for (int i = 1; i < n; i++) {
        if (dst[i] != 0x10 + i || old[i] != i - 1 + m)
          abort();
      }
    }

    if (__atomic_exchange_1(&a8, U8(V + m + 1), model) != U8(V + m))
      abort();
    if (__atomic_exchange_2(&a16, U16(V + m + 1), model) != U16(V + m))
      abort();
    if (__atomic_exchange_4(&a32, U32(V + m + 1), model) != U32(V + m))
      abort();
    if (__atomic_exchange_8(&a64, U64(V + m + 1), model) != U64(V + m))
      abort();
#ifdef TEST_16
    if (__atomic_exchange_16(&a128, V + m + 1, model) != V + m)
      abort();
#endif
  }
}

void test_compare_exchanges(void) {
  static int atomic_compare_exchange_models[] = {
      __ATOMIC_RELAXED,
      __ATOMIC_CONSUME,
      __ATOMIC_ACQUIRE,
      __ATOMIC_SEQ_CST,
      __ATOMIC_RELEASE,
      __ATOMIC_ACQ_REL,
  };

  for (int m1 = 0; m1 < LEN(atomic_compare_exchange_models); m1++) {
    // Skip the last two: __ATOMIC_RELEASE and __ATOMIC_ACQ_REL.
    // See <http://wg21.link/p0418> for details.
    for (int m2 = 0; m2 < LEN(atomic_compare_exchange_models) - 2; m2++) {
      int m_succ = atomic_compare_exchange_models[m1];
      int m_fail = atomic_compare_exchange_models[m2];

      // Test with aligned data.
      for (int n = 1; n <= LEN(data); n++) {
        __attribute__((aligned(16))) char dst[LEN(data)] = {0};
        __attribute__((aligned(16))) char exp[LEN(data)] = {0};
        if (!__atomic_compare_exchange_c(n, dst, exp, data, m_succ, m_fail))
          abort();
        if (memcmp(dst, data, n) != 0)
          abort();
        if (__atomic_compare_exchange_c(n, dst, exp, data, m_succ, m_fail))
          abort();
        if (memcmp(exp, data, n) != 0)
          abort();
      }

      // Test with unaligned data.
      for (int n = 1; n < LEN(data); n++) {
        __attribute__((aligned(16))) char dst[LEN(data)] = {0};
        __attribute__((aligned(16))) char exp[LEN(data)] = {0};
        if (!__atomic_compare_exchange_c(n, dst + 1, exp + 1, data + 1,
                                         m_succ, m_fail))
          abort();
        if (memcmp(dst + 1, data + 1, n) != 0)
          abort();
        if (__atomic_compare_exchange_c(n, dst + 1, exp + 1, data + 1, m_succ,
                                        m_fail))
          abort();
        if (memcmp(exp + 1, data + 1, n) != 0)
          abort();
      }

      set_a_values(ONES);
      set_b_values(ONES * 2);

      if (__atomic_compare_exchange_1(&a8, &b8, U8(V + m1), m_succ, m_fail))
        abort();
      if (a8 != U8(ONES) || b8 != U8(ONES))
        abort();
      if (!__atomic_compare_exchange_1(&a8, &b8, U8(V + m1), m_succ, m_fail))
        abort();
      if (a8 != U8(V + m1) || b8 != U8(ONES))
        abort();

      if (__atomic_compare_exchange_2(&a16, &b16, U16(V + m1), m_succ, m_fail))
        abort();
      if (a16 != U16(ONES) || b16 != U16(ONES))
        abort();
      if (!__atomic_compare_exchange_2(&a16, &b16, U16(V + m1), m_succ, m_fail))
        abort();
      if (a16 != U16(V + m1) || b16 != U16(ONES))
        abort();

      if (__atomic_compare_exchange_4(&a32, &b32, U32(V + m1), m_succ, m_fail))
        abort();
      if (a32 != U32(ONES) || b32 != U32(ONES))
        abort();
      if (!__atomic_compare_exchange_4(&a32, &b32, U32(V + m1), m_succ, m_fail))
        abort();
      if (a32 != U32(V + m1) || b32 != U32(ONES))
        abort();

      if (__atomic_compare_exchange_8(&a64, &b64, U64(V + m1), m_succ, m_fail))
        abort();
      if (a64 != U64(ONES) || b64 != U64(ONES))
        abort();
      if (!__atomic_compare_exchange_8(&a64, &b64, U64(V + m1), m_succ, m_fail))
        abort();
      if (a64 != U64(V + m1) || b64 != U64(ONES))
        abort();

#ifdef TEST_16
      if (__atomic_compare_exchange_16(&a128, &b128, V + m1, m_succ, m_fail))
        abort();
      if (a128 != ONES || b128 != ONES)
        abort();
      if (!__atomic_compare_exchange_16(&a128, &b128, V + m1, m_succ, m_fail))
        abort();
      if (a128 != V + m1 || b128 != ONES)
        abort();
#endif
    }
  }
}

void test_fetch_op(void) {
  static int atomic_fetch_models[] = {
      __ATOMIC_RELAXED,
      __ATOMIC_CONSUME,
      __ATOMIC_ACQUIRE,
      __ATOMIC_RELEASE,
      __ATOMIC_ACQ_REL,
      __ATOMIC_SEQ_CST,
  };

  for (int m = 0; m < LEN(atomic_fetch_models); m++) {
    int model = atomic_fetch_models[m];

    // Fetch add.

    set_a_values(V + m);
    set_b_values(0);
    b8 = __atomic_fetch_add_1(&a8, U8(ONES), model);
    if (b8 != U8(V + m) || a8 != U8(V + m + ONES))
      abort();
    b16 = __atomic_fetch_add_2(&a16, U16(ONES), model);
    if (b16 != U16(V + m) || a16 != U16(V + m + ONES))
      abort();
    b32 = __atomic_fetch_add_4(&a32, U32(ONES), model);
    if (b32 != U32(V + m) || a32 != U32(V + m + ONES))
      abort();
    b64 = __atomic_fetch_add_8(&a64, U64(ONES), model);
    if (b64 != U64(V + m) || a64 != U64(V + m + ONES))
      abort();
#ifdef TEST_16
    b128 = __atomic_fetch_add_16(&a128, ONES, model);
    if (b128 != V + m || a128 != V + m + ONES)
      abort();
#endif

    // Fetch sub.

    set_a_values(V + m);
    set_b_values(0);
    b8 = __atomic_fetch_sub_1(&a8, U8(ONES), model);
    if (b8 != U8(V + m) || a8 != U8(V + m - ONES))
      abort();
    b16 = __atomic_fetch_sub_2(&a16, U16(ONES), model);
    if (b16 != U16(V + m) || a16 != U16(V + m - ONES))
      abort();
    b32 = __atomic_fetch_sub_4(&a32, U32(ONES), model);
    if (b32 != U32(V + m) || a32 != U32(V + m - ONES))
      abort();
    b64 = __atomic_fetch_sub_8(&a64, U64(ONES), model);
    if (b64 != U64(V + m) || a64 != U64(V + m - ONES))
      abort();
#ifdef TEST_16
    b128 = __atomic_fetch_sub_16(&a128, ONES, model);
    if (b128 != V + m || a128 != V + m - ONES)
      abort();
#endif

    // Fetch and.

    set_a_values(V + m);
    set_b_values(0);
    b8 = __atomic_fetch_and_1(&a8, U8(V + m), model);
    if (b8 != U8(V + m) || a8 != U8(V + m))
      abort();
    b16 = __atomic_fetch_and_2(&a16, U16(V + m), model);
    if (b16 != U16(V + m) || a16 != U16(V + m))
      abort();
    b32 = __atomic_fetch_and_4(&a32, U32(V + m), model);
    if (b32 != U32(V + m) || a32 != U32(V + m))
      abort();
    b64 = __atomic_fetch_and_8(&a64, U64(V + m), model);
    if (b64 != U64(V + m) || a64 != U64(V + m))
      abort();
#ifdef TEST_16
    b128 = __atomic_fetch_and_16(&a128, V + m, model);
    if (b128 != V + m || a128 != V + m)
      abort();
#endif

    // Fetch or.

    set_a_values(V + m);
    set_b_values(0);
    b8 = __atomic_fetch_or_1(&a8, U8(ONES), model);
    if (b8 != U8(V + m) || a8 != U8((V + m) | ONES))
      abort();
    b16 = __atomic_fetch_or_2(&a16, U16(ONES), model);
    if (b16 != U16(V + m) || a16 != U16((V + m) | ONES))
      abort();
    b32 = __atomic_fetch_or_4(&a32, U32(ONES), model);
    if (b32 != U32(V + m) || a32 != U32((V + m) | ONES))
      abort();
    b64 = __atomic_fetch_or_8(&a64, U64(ONES), model);
    if (b64 != U64(V + m) || a64 != U64((V + m) | ONES))
      abort();
#ifdef TEST_16
    b128 = __atomic_fetch_or_16(&a128, ONES, model);
    if (b128 != V + m || a128 != ((V + m) | ONES))
      abort();
#endif

    // Fetch xor.

    set_a_values(V + m);
    set_b_values(0);
    b8 = __atomic_fetch_xor_1(&a8, U8(ONES), model);
    if (b8 != U8(V + m) || a8 != U8((V + m) ^ ONES))
      abort();
    b16 = __atomic_fetch_xor_2(&a16, U16(ONES), model);
    if (b16 != U16(V + m) || a16 != U16((V + m) ^ ONES))
      abort();
    b32 = __atomic_fetch_xor_4(&a32, U32(ONES), model);
    if (b32 != U32(V + m) || a32 != U32((V + m) ^ ONES))
      abort();
    b64 = __atomic_fetch_xor_8(&a64, U64(ONES), model);
    if (b64 != U64(V + m) || a64 != U64((V + m) ^ ONES))
      abort();
#ifdef TEST_16
    b128 = __atomic_fetch_xor_16(&a128, ONES, model);
    if (b128 != (V + m) || a128 != ((V + m) ^ ONES))
      abort();
#endif

    // Check signed integer overflow behavior

    set_a_values(V + m);
    __atomic_fetch_add_1(&a8, U8(V), model);
    if (a8 != U8(V * 2 + m))
      abort();
    __atomic_fetch_sub_1(&a8, U8(V), model);
    if (a8 != U8(V + m))
      abort();
    __atomic_fetch_add_2(&a16, U16(V), model);
    if (a16 != U16(V * 2 + m))
      abort();
    __atomic_fetch_sub_2(&a16, U16(V), model);
    if (a16 != U16(V + m))
      abort();
    __atomic_fetch_add_4(&a32, U32(V), model);
    if (a32 != U32(V * 2 + m))
      abort();
    __atomic_fetch_sub_4(&a32, U32(V), model);
    if (a32 != U32(V + m))
      abort();
    __atomic_fetch_add_8(&a64, U64(V), model);
    if (a64 != U64(V * 2 + m))
      abort();
    __atomic_fetch_sub_8(&a64, U64(V), model);
    if (a64 != U64(V + m))
      abort();
#ifdef TEST_16
    __atomic_fetch_add_16(&a128, V, model);
    if (a128 != V * 2 + m)
      abort();
    __atomic_fetch_sub_16(&a128, V, model);
    if (a128 != V + m)
      abort();
#endif
  }
}

int main() {
  test_loads();
  test_stores();
  test_exchanges();
  test_compare_exchanges();
  test_fetch_op();
  return 0;
}
