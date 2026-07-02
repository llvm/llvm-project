// RUN: %clang_builtins -std=c23 %s %librt -o %t && %run %t
// REQUIRES: librt_has_atomic
//===-- atomic_bitint_test.c - Test atomic ops on _BitInt -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime checks for atomic read-modify-write on _BitInt(N). A padded width
// (37) exercises the inline compare-exchange loop; a wide width (256) exercises
// the __atomic_compare_exchange libcall loop. Each op is cross-checked against
// the same operation done non-atomically, and the dirty-padding cases confirm
// the loop converges (a re-canonicalized expected would spin forever).
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef signed _BitInt(37) S37;
typedef unsigned _BitInt(37) U37;
typedef signed _BitInt(256) S256; // no padding (exactly 32 bytes)
typedef signed _BitInt(200) S200; // padded: 200 value bits in 32-byte storage

// Each macro runs the atomic op and asserts the returned old value and the
// resulting object both match the non-atomic computation at width N.
#define CHECK_FETCH(T, init, op, rhs, expr)                                    \
  do {                                                                         \
    _Atomic(T) a = (init);                                                     \
    T old = __c11_atomic_fetch_##op(&a, (rhs), __ATOMIC_SEQ_CST);              \
    assert(old == (T)(init));                                                  \
    assert((T)a == (T)(expr));                                                 \
  } while (0)

static void test_ops(void) {
  CHECK_FETCH(S37, 100, add, 5, 105);
  CHECK_FETCH(S37, 100, sub, 40, 60);
  CHECK_FETCH(S37, -3, add, 1, -2);
  CHECK_FETCH(U37, 7, add, 9, 16);
  CHECK_FETCH(S37, 0x15, and, 0x13, 0x11);
  CHECK_FETCH(S37, 0x10, or, 5, 0x15);
  CHECK_FETCH(S37, 0x1F, xor, 0x15, 0x0A);
  CHECK_FETCH(S37, -5, min, -7, -7);    // signed: -7 < -5
  CHECK_FETCH(U37, 5, min, (U37)-1, 5); // unsigned: 5 < 2^37-1
  CHECK_FETCH(S37, 3, max, 9, 9);
  CHECK_FETCH(S37, 0x15, nand, 0x13, (S37) ~(0x15 & 0x13));
  // Wide widths: the libcall loop (no padding, and padded).
  CHECK_FETCH(S256, 100, add, 5, 105);
  CHECK_FETCH(S256, 1, or, 0xFE, 0xFF);
  CHECK_FETCH(S200, 100, add, 5, 105);
}

// Seed non-canonical padding through a union, then RMW. A loop that carried a
// re-canonicalized expected would never match memory and hang here.
static void test_dirty_padding(void) {
  // uint64_t (not unsigned long, which is 32-bit on LLP64) so the padding bit
  // is representable and the overlay matches the 8-byte atomic.
  union {
    _Atomic(S37) a;
    uint64_t b;
  } s;
  s.b = ((uint64_t)1 << 40) | 5u; // value bits 5, padding bit 40 set
  S37 old = __c11_atomic_fetch_add(&s.a, 1, __ATOMIC_SEQ_CST);
  assert(old == 5 && (S37)s.a == 6);
  assert((s.b >> 37) == 0); // padding canonicalized (positive value)

  union {
    _Atomic(U37) a;
    uint64_t b;
  } u;
  u.b = ((uint64_t)3 << 50) | 7u;
  U37 uold = __c11_atomic_fetch_add(&u.a, 1, __ATOMIC_SEQ_CST);
  assert(uold == 7 && (U37)u.a == 8);
  assert((u.b >> 37) == 0); // padding canonicalized (zero-extended)

  // Wide padded width (libcall loop): _BitInt(200) has 56 padding bits in its
  // 32-byte storage. Set the overlay at value level (endian-independent): low
  // 200 bits = 5, a padding bit (240) dirtied.
  union {
    _Atomic(S200) a;
    unsigned _BitInt(256) full;
  } w;
  w.full = (unsigned _BitInt(256))5 | ((unsigned _BitInt(256))0xAA << 240);
  S200 wold = __c11_atomic_fetch_add(&w.a, 1, __ATOMIC_SEQ_CST);
  assert(wold == 5 && (S200)w.a == 6);
  assert((w.full >> 200) == 0); // padding canonicalized (positive value)
}

// The _fetch builtins return the new value, not the old one.
static void test_returns_new(void) {
  S37 a = 100;
  assert(__atomic_add_fetch(&a, 5, __ATOMIC_SEQ_CST) == 105);
  assert(__atomic_sub_fetch(&a, 10, __ATOMIC_SEQ_CST) == 95);
  U37 u = 0;
  assert(__atomic_or_fetch(&u, 0xF, __ATOMIC_SEQ_CST) == 0xF);
  S200 w = 100;
  assert(__atomic_add_fetch(&w, 5, __ATOMIC_SEQ_CST) == 105);
}

// Each non-seq_cst ordering drives the loop's load/cmpxchg ordering.
static void test_orderings(void) {
  _Atomic(S37) a = 10;
  (void)__c11_atomic_fetch_add(&a, 1, __ATOMIC_RELAXED);
  (void)__c11_atomic_fetch_add(&a, 1, __ATOMIC_ACQUIRE);
  (void)__c11_atomic_fetch_add(&a, 1, __ATOMIC_RELEASE);
  (void)__c11_atomic_fetch_add(&a, 1, __ATOMIC_ACQ_REL);
  assert((S37)a == 14);
}

int main(void) {
  test_ops();
  test_dirty_padding();
  test_returns_new();
  test_orderings();
  printf("PASS\n");
  return 0;
}
