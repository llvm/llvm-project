//===-- udivmodei5.c - Implement _udivmodei5-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements _udivmodei5 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

// When this is built into the bitint library, provide symbols with
// weak linkage to allow gracefully upgrading once libgcc contains those
// symbols.
#ifdef IS_BITINT_LIBRARY
#define WEAK_IF_BITINT_LIBRARY __attribute__((weak))
#else
#define WEAK_IF_BITINT_LIBRARY
#endif

static const int WORD_SIZE_IN_BITS = sizeof(su_int) * CHAR_BIT;

/// A mask with the most significant bit set.
static const su_int WORD_MSB = (su_int)1 << (WORD_SIZE_IN_BITS - 1);

// Define an index such that a[WORD_IDX(0)] is the least significant word
// and a[WORD_IDX(words-1)] is the most significant word.
#if _YUGA_LITTLE_ENDIAN
#define WORD_IDX(X, words) (X)
#else
#define WORD_IDX(X, words) ((words) - (X))
#endif

static bool has_msb_set(su_int a) { return (a & WORD_MSB) != 0; }

static bool is_neg(const uint32_t *V, unsigned words) {
  return has_msb_set(V[WORD_IDX(words - 1, words)]);
}

/// dst = ~dst
static void complement(su_int *dst, unsigned words) {
  for (unsigned i = 0; i < words; i++)
    dst[i] = ~dst[i];
}

/// dst += src
static void add_part(su_int *dst, su_int src, unsigned words) {
  for (unsigned i = 0; i < words; ++i) {
    dst[WORD_IDX(i, words)] += src;
    if (dst[WORD_IDX(i, words)] >= src)
      return; // No carry
    src = 1;
  }
}

/// dst += 1
static void increment(su_int *dst, unsigned words) { add_part(dst, 1u, words); }

/// dst = -dst
static void negate(su_int *dst, unsigned words) {
  complement(dst, words);
  increment(dst, words);
}

/// a -= b
/// \pre a >= b
static void subtract(su_int *a, const su_int *b, unsigned words) {
  su_int carry = 0;
  for (unsigned i = 0; i < words; ++i) {
    su_int dst = 0;
    carry = __builtin_sub_overflow(a[WORD_IDX(i, words)], carry, &dst);
    carry += __builtin_sub_overflow(dst, b[WORD_IDX(i, words)],
                                    &a[WORD_IDX(i, words)]);
  }
}

/// a = 0
static void set_zero(su_int *a, unsigned words) {
  for (unsigned i = 0; i < words; ++i)
    a[i] = 0;
}

/// a > b: return +1
/// a < b: return -1
/// a == b: return 0
static int ucompare(const su_int *a, const su_int *b, unsigned int words) {
  for (int i = words - 1; i >= 0; --i) {
    if (a[WORD_IDX(i, words)] != b[WORD_IDX(i, words)])
      return a[WORD_IDX(i, words)] > b[WORD_IDX(i, words)] ? 1 : -1;
  }
  return 0;
}

/// Performs a logic left shift of one bit of 'a'.
static void left_shift_1(su_int *a, unsigned words) {
  for (int i = words - 1; i >= 0; --i) {
    a[WORD_IDX(i, words)] <<= 1;
    if (i == 0)
      continue;
    if (has_msb_set(a[WORD_IDX(i - 1, words)]))
      a[WORD_IDX(i, words)] |= 1;
  }
}

/// Set the least signitificant bit of a to 'a'.
/// \pre The least signitificant bit of 'a' is zero.
static void set_bit_0(su_int *a, unsigned words, unsigned v) {
  a[WORD_IDX(0, words)] |= v;
}

/// Sets the n-th bit of 'a' to 1.
/// \pre The n-th bit of 'a' is zero.
static void set_bit(su_int *a, unsigned words, unsigned n) {
  unsigned word = n / WORD_SIZE_IN_BITS;
  unsigned bit = n % WORD_SIZE_IN_BITS;
  a[WORD_IDX(word, words)] |= (su_int)1 << bit;
}

/// Returns the n-th bit of 'a'.
static unsigned get_bit(const su_int *a, unsigned words, unsigned n) {
  unsigned word = n / WORD_SIZE_IN_BITS;
  unsigned bit = n % WORD_SIZE_IN_BITS;
  su_int mask = (su_int)1 << bit;
  return !!(a[WORD_IDX(word, words)] & mask);
}

/// Unsigned divison quo = a / b, rem = a % b
WEAK_IF_BITINT_LIBRARY
COMPILER_RT_ABI void __udivmodei5(su_int *quo, su_int *rem, su_int *a,
                                  su_int *b, unsigned int words) {

  set_zero(quo, words);
  set_zero(rem, words);

  unsigned bits = words * WORD_SIZE_IN_BITS;
  for (int i = bits - 1; i >= 0; --i) {
    left_shift_1(rem, words);                    // rem <<= 1;
    set_bit_0(rem, words, get_bit(a, words, i)); // rem(bit 0) = a(bit i);
    if (ucompare(rem, b, words) >= 0) {          // if (rem >= b)
      subtract(rem, b, words);                   // rem -= b;
      set_bit(quo, words, i);                    // quo(bit i) = 1;
    }
  }
}

/// Signed divison quo = a / b, rem = a % b
WEAK_IF_BITINT_LIBRARY
COMPILER_RT_ABI void __divmodei5(su_int *quo, su_int *rem, su_int *a, su_int *b,
                                 unsigned int words) {
  int asign = is_neg(a, words);
  int bsign = is_neg(b, words);
  if (asign) {
    if (bsign) {
      negate(a, words);
      negate(b, words);
      __udivmodei5(quo, rem, a, b, words);
    } else {
      negate(a, words);
      __udivmodei5(quo, rem, a, b, words);
      negate(quo, words);
    }
    negate(rem, words);
  } else if (bsign) {
    negate(b, words);
    __udivmodei5(quo, rem, a, b, words);
    negate(quo, words);
  } else {
    __udivmodei5(quo, rem, a, b, words);
  }
}
