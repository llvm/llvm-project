//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: librt_has_outline_atomic_cas16_1, int128
// RUN: %clang_builtins %s %librt -o %t && %run %t

// Set by cpu_model/aarch64.c; lse.S uses its LL/SC fallback when it is false.
#if !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
extern _Bool __aarch64_have_lse_atomics;

// lse.S defines these helpers without the user label prefix that C names get
// on Mach-O, so name the symbols explicitly.
typedef __uint128_t cas16_fn(__uint128_t expected, __uint128_t desired,
                             __uint128_t *ptr);
cas16_fn __aarch64_cas16_relax __asm__("__aarch64_cas16_relax");
cas16_fn __aarch64_cas16_acq __asm__("__aarch64_cas16_acq");
cas16_fn __aarch64_cas16_rel __asm__("__aarch64_cas16_rel");
cas16_fn __aarch64_cas16_acq_rel __asm__("__aarch64_cas16_acq_rel");
cas16_fn __aarch64_cas16_sync __asm__("__aarch64_cas16_sync");

static int test_cas16(void) {
  __uint128_t val;
  __uint128_t expected;
  __uint128_t desired;
  __uint128_t old;

  // Successful exchanges, using each memory order in turn.
  val = ((__uint128_t)0x0123456789abcdefULL << 64) | 0xfedcba9876543210ULL;
  expected = val;
  desired = ((__uint128_t)0x1111111111111111ULL << 64) | 0x2222222222222222ULL;
  old = __aarch64_cas16_relax(expected, desired, &val);
  if (old != expected || val != desired)
    return 1;

  expected = val;
  desired = ((__uint128_t)0x3333333333333333ULL << 64) | 0x4444444444444444ULL;
  old = __aarch64_cas16_acq(expected, desired, &val);
  if (old != expected || val != desired)
    return 1;

  expected = val;
  desired = ((__uint128_t)0x5555555555555555ULL << 64) | 0x6666666666666666ULL;
  old = __aarch64_cas16_rel(expected, desired, &val);
  if (old != expected || val != desired)
    return 1;

  expected = val;
  desired = ((__uint128_t)0x7777777777777777ULL << 64) | 0x8888888888888888ULL;
  old = __aarch64_cas16_acq_rel(expected, desired, &val);
  if (old != expected || val != desired)
    return 1;

  expected = val;
  desired = ((__uint128_t)0x9999999999999999ULL << 64) | 0xaaaaaaaaaaaaaaaaULL;
  old = __aarch64_cas16_sync(expected, desired, &val);
  if (old != expected || val != desired)
    return 1;

  // Failed exchanges where only the low 64 bits differ.
  __uint128_t current_val = val;
  expected = current_val ^ 0x1ULL;
  desired = ((__uint128_t)0xbbbbbbbbbbbbbbbbULL << 64) | 0xccccccccccccccccULL;
  old = __aarch64_cas16_relax(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_acq(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_rel(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_acq_rel(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_sync(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  // Failed exchanges where only the high 64 bits differ.
  expected = current_val ^ ((__uint128_t)0x1ULL << 64);
  desired = ((__uint128_t)0xddddddddddddddddULL << 64) | 0xeeeeeeeeeeeeeeeeULL;
  old = __aarch64_cas16_relax(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_acq(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_rel(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_acq_rel(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  old = __aarch64_cas16_sync(expected, desired, &val);
  if (old != current_val || val != current_val)
    return 1;

  return 0;
}

int main(void) {
  _Bool has_lse = __aarch64_have_lse_atomics;

  // Test the LL/SC fallback path even if the CPU has LSE.
  __aarch64_have_lse_atomics = 0;
  if (test_cas16())
    return 1;

  // Also test the CASP path if the CPU has LSE.
  if (has_lse) {
    __aarch64_have_lse_atomics = 1;
    if (test_cas16())
      return 1;
  }

  return 0;
}
