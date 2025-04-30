/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Implement 128-bit integer operations
 *
 *  Wraps GCC's __int128 with portable interfaces when __int128
 *  is available; implements 128-bit integer arithmetic with 64-bit
 *  operations when it is not.
 */

#include "int128.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8)

/*
 *  Use the C compiler's native __int128 type when its available.
 */

static const int128_t zero = 0;
static const int128_t one = 1;

void
int128_from_uint64(int128_t *result, uint64_t x)
{
  *result = x;
}

void
int128_from_int64(int128_t *result, int64_t x)
{
  *result = x;
}

bool
int128_to_uint64(uint64_t *result, const int128_t *x)
{
  *result = *x;
  return *result != *x;
}

bool
int128_to_int64(int64_t *result, const int128_t *x)
{
  *result = *x;
  return *result != *x;
}

int
int128_unsigned_compare(const int128_t *x, const int128_t *y)
{
  uint128_t ux = *x, uy = *y;
  if (ux < uy)
    return -1;
  return ux > uy;
}

int
int128_signed_compare(const int128_t *x, const int128_t *y)
{
  if (*x < *y)
    return -1;
  return *x > *y;
}

int
int128_count_leading_zeros(const int128_t *x)
{
  int128_t v = *x;
  int128_t mask = 1;
  mask <<= 127;
  int j;
  for(j = 0; j < 128 && 0 == (mask & v); j++) {
    mask >>= 1;
  }
  return j;
}

void
int128_ones_complement(int128_t *result, const int128_t *x)
{
  *result = ~*x;
}

bool
int128_twos_complement(int128_t *result, const int128_t *x)
{
  *result = -*x;
  return *x != 0 && *result == *x;
}

void
int128_and(int128_t *result, const int128_t *x, const int128_t *y)
{
  *result = *x & *y;
}

void
int128_or(int128_t *result, const int128_t *x, const int128_t *y)
{
  *result = *x | *y;
}

void
int128_xor(int128_t *result, const int128_t *x, const int128_t *y)
{
  *result = *x ^ *y;
}

void
int128_shift_left(int128_t *result, const int128_t *x, int count)
{
  assert(count >= 0);
  *result = *x << count;
}

void
int128_shift_right_logical(int128_t *result, const int128_t *x, int count)
{
  assert(count >= 0);
  *result = (uint128_t) *x >> count;
}

bool
int128_unsigned_add(int128_t *result, const int128_t *x, const int128_t *y)
{
  const uint128_t *ux = x, *uy = y;
  uint128_t *uresult = result;
  *uresult = *uy + *ux;
  return *uresult < *ux || *uresult < *uy;
}

bool
int128_signed_add(int128_t *result, const int128_t *x, const int128_t *y)
{
  bool xs = *x < 0, ys = *y < 0;
  *result = *x + *y;
  return xs == ys && xs != (*result < 0);
}

bool
int128_signed_subtract(int128_t *result, const int128_t *x, const int128_t *y)
{
  int128_t tmp = -*y;
  return int128_signed_add(result, x, &tmp);
}

void
int128_unsigned_multiply(int128_t *high, int128_t *low,
                         const int128_t *x, const int128_t *y)
{
  uint128_t xh = *x >> 64, xl = (uint64_t) *x;
  uint128_t yh = *y >> 64, yl = (uint64_t) *y;
  *low = *x * *y;
  *high = ((((xl * yl) >> 64) + (xh * yl) + (yh * xl)) >> 64) + xh * yh;
}

#else /* no native __int128, so we synthesize it with a struct. */

#define MSB32 0x80000000

static const int128_t zero = { { 0, 0, 0, 0 } };
static const int128_t one = { { 1, 0, 0, 0 } };

void
int128_from_uint64(int128_t *result, uint64_t x)
{
  result->part[0] = x;
  result->part[1] = x >> 32;
  result->part[2] = result->part[3] = 0;
}

void
int128_from_int64(int128_t *result, int64_t x)
{
  result->part[0] = x;
  result->part[1] = x >> 32;
  result->part[2] = result->part[3] = -(x < 0);
}

bool
int128_to_uint64(uint64_t *result, const int128_t *x)
{
  *result = ((uint64_t) x->part[1] << 32) | x->part[0];
  return (x->part[2] | x->part[3]) != 0;
}

bool
int128_to_int64(int64_t *result, const int128_t *x)
{
  uint32_t sext = -((int32_t) x->part[1] < 0);
  *result = ((uint64_t) x->part[1] << 32) | x->part[0];
  return (x->part[2] != sext) | (x->part[3] != sext);
}

int
int128_unsigned_compare(const int128_t *x, const int128_t *y)
{
  int j;
  for (j = 3; j >= 0; --j) {
    if (x->part[j] < y->part[j])
      return -1;
    if (x->part[j] > y->part[j])
      return 1;
  }
  return 0;
}

int
int128_signed_compare(const int128_t *x, const int128_t *y)
{
  if (x->part[3] & MSB32) {
    if (!(y->part[3] & MSB32))
      return -1;
    return -int128_unsigned_compare(x, y);
  } else {
    if (y->part[3] & MSB32)
      return 1;
    return int128_unsigned_compare(x, y);
  }
}

int
int128_count_leading_zeros(const int128_t *x)
{
  int j, k;
  for (j = 0; j < 4; ++j) {
    uint32_t w = x->part[3 - j];
    if (w != 0) {
      for (k = 0; k < 32; ++k) {
        if (((w << k) & MSB32) != 0)
          return 32 * j + k;
      }
      assert(!"int128_count_leading_zeros: can't happen");
    }
  }
  return 128;
}

void
int128_ones_complement(int128_t *result, const int128_t *x)
{
  int j;
  for (j = 0; j < 4; ++j) {
    result->part[j] = ~x->part[j];
  }
}

bool
int128_twos_complement(int128_t *result, const int128_t *x)
{
  int128_ones_complement(result, x);
  return int128_signed_add(result, result, &one);
}

void
int128_and(int128_t *result, const int128_t *x, const int128_t *y)
{
  int j;
  for (j = 0; j < 4; ++j) {
    result->part[j] = x->part[j] & y->part[j];
  }
}

void
int128_or(int128_t *result, const int128_t *x, const int128_t *y)
{
  int j;
  for (j = 0; j < 4; ++j) {
    result->part[j] = x->part[j] | y->part[j];
  }
}

void
int128_xor(int128_t *result, const int128_t *x, const int128_t *y)
{
  int j;
  for (j = 0; j < 4; ++j) {
    result->part[j] = x->part[j] ^ y->part[j];
  }
}

void
int128_shift_left(int128_t *result, const int128_t *x, int count)
{
  int j, off, sh;
  assert(count >= 0);
  off = count >> 5;
  sh = count & 31;
  if (sh == 0) {
    /* Be portable: don't assume that a shift by (32-0) below yields 0. */
    for (j = 3; j - off >= 0; --j) {
      result->part[j] = x->part[j - off];
    }
  } else {
    for (j = 3; j - off - 1 >= 0; --j) {
      result->part[j] = (x->part[j - off] << sh) |
                        (x->part[j - off - 1] >> (32 - sh));
    }
    if (j - off >= 0) {
      result->part[j] = x->part[j - off] << sh;
      --j;
    }
  }
  for (; j >= 0; --j) {
    result->part[j] = 0;
  }
}

void
int128_shift_right_logical(int128_t *result, const int128_t *x, int count)
{
  int j, off, sh;
  assert(count >= 0);
  off = count >> 5;
  sh = count & 31;
  if (sh == 0) {
    /* Be portable: don't assume that a shift by (32-0) below yields 0. */
    for (j = 0; j + off < 4; ++j) {
      result->part[j] = x->part[j + off];
    }
  } else {
    for (j = 0; j + off + 1 < 4; ++j) {
      result->part[j] = (x->part[j + off] >> sh) |
                        (x->part[j + off + 1] << (32 - sh));
    }
    if (j + off < 4) {
      result->part[j] = x->part[j + off] >> sh;
      ++j;
    }
  }
  for (; j < 4; ++j) {
    result->part[j] = 0;
  }
}

bool
int128_unsigned_add(int128_t *result, const int128_t *x, const int128_t *y)
{
  uint64_t carry = 0;
  int j;
  for (j = 0; j < 4; ++j) {
    carry += x->part[j];
    result->part[j] = carry += y->part[j];
    carry >>= 32;
  }
  return carry != 0;
}

bool
int128_signed_add(int128_t *result, const int128_t *x, const int128_t *y)
{
  bool unsigned_carry = int128_unsigned_add(result, x, y);
  return unsigned_carry != ((result->part[3] & MSB32) != 0);
}

bool
int128_signed_subtract(int128_t *result, const int128_t *x, const int128_t *y)
{
  int128_t tmp;
  int128_twos_complement(&tmp, y);
  return int128_signed_add(result, x, &tmp);
}

void
int128_unsigned_multiply(int128_t *high, int128_t *low,
                         const int128_t *x, const int128_t *y)
{
  int j, k, to;
  uint32_t res[8];
  for (j = 0; j < 8; ++j) {
    res[j] = 0;
  }
  for (j = 0; j < 4; ++j) {
    for (k = 0; k < 4; ++k) {
      /* Next few expressions can't overflow, since they compute at most
       *    0xffffffff * 0xffffffff + 0xffffffff
       *  = (0xffffffff + 1) * 0xffffffff
       *  = 0x100000000 * 0xffffffff
       *  = 0xffffffff00000000
       * or, if you prefer,
       *    (2**32 - 1)**2 + 2**32 - 1
       *  = 2**64 - 2*(2**32) + 1 + 2**32 - 1
       *  = 2**64 - 2**32
       */
      uint64_t xy = (uint64_t) x->part[j] * y->part[k];
      for (to = j + k; to < 8; ++to) {
        if (xy == 0)
          break;
        res[to] = xy += res[to];
        xy >>= 32;
      }
    }
  }
  for (j = 0; j < 4; ++j) {
    low->part[j] = res[j];
    high->part[j] = res[j + 4];
  }
}
#endif

/*
 *  These routines are independent of the implementation of int128_t.
 */

void
int128_signed_multiply(int128_t *high, int128_t *low,
                       const int128_t *x, const int128_t *y)
{
  bool xneg = int128_signed_compare(x, &zero) < 0;
  bool yneg = int128_signed_compare(y, &zero) < 0;
  int128_t xtmp, ytmp;

  if (xneg) {
    int128_twos_complement(&xtmp, x);
    x = &xtmp;
  }
  if (yneg) {
    int128_twos_complement(&ytmp, y);
    y = &ytmp;
  }
  int128_unsigned_multiply(high, low, x, y);
  if (xneg != yneg) {
    int128_ones_complement(low, low);
    int128_ones_complement(high, high);
    if (int128_unsigned_add(low, low, &one))
      int128_unsigned_add(high, high, &one);
  }
}

bool
int128_unsigned_divide(int128_t *quotient, int128_t *remainder,
                       const int128_t *dividend, const int128_t *divisor)
{
  bool allzero = int128_unsigned_compare(divisor, &zero) == 0;
  int128_t top;
  int bits = int128_count_leading_zeros(dividend);

  *quotient = zero;
  *remainder = zero;
  int128_shift_left(&top, dividend, bits);
  for (; bits < 128; ++bits) {
    int128_t tmp = *remainder;
    int128_unsigned_add(remainder, &tmp, &tmp);
    tmp = top;
    if (int128_unsigned_add(&top, &tmp, &tmp)) {
      tmp = *remainder;
      int128_unsigned_add(remainder, &tmp, &one);
    }
    tmp = *quotient;
    int128_unsigned_add(quotient, &tmp, &tmp);
    if (int128_unsigned_compare(remainder, divisor) >= 0) {
      tmp = *quotient;
      int128_unsigned_add(quotient, &tmp, &one);
      tmp = *remainder;
      int128_signed_subtract(remainder, &tmp, divisor);
    }
  }
  return allzero;
}

bool
int128_signed_divide(int128_t *quotient, int128_t *remainder,
                     const int128_t *dividend, const int128_t *divisor)
{
  bool negate_quotient = false, negate_remainder = false;
  bool overflow = false;
  int128_t tmp[4];
  const int128_t *num = dividend, *denom = divisor;
  if (int128_signed_compare(dividend, &zero) < 0) {
    negate_quotient = negate_remainder = true;
    int128_twos_complement(&tmp[0], num);
    num = &tmp[0];
  }
  if (int128_signed_compare(divisor, &zero) < 0) {
    negate_quotient = !negate_quotient;
    int128_twos_complement(&tmp[1], denom);
    denom = &tmp[1];
  }
  overflow = int128_unsigned_divide(quotient, remainder, num, denom);
  if (int128_signed_compare(quotient, &zero) < 0) {
    /* Signed division can overflow in one case: MOST_NEGATIVE_INT / -1 */
    overflow = true;
  }
  if (negate_quotient) {
    tmp[2] = *quotient;
    int128_twos_complement(quotient, &tmp[2]);
  }
  if (negate_remainder) {
    tmp[3] = *remainder;
    int128_twos_complement(remainder, &tmp[3]);
  }
  return overflow;
}
