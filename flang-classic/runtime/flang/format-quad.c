/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "float128.h"
#include "format-double.h"
#include <string.h>
#ifndef TARGET_WIN
#include <stdbool.h>
#include <stdint.h>
#else
typedef enum bool { false, true } bool;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif

typedef __uint128_t uint128_t;

/*
 *  IEEE-754 quad precision parameters
 */
#define EXPONENT_FIELD_BITS 15
#define EXPLICIT_MANTISSA_BITS 112
#define EXPONENT_BIAS 0x3ffe /* i.e., exponent field value of 0.5 */
#define INF_OR_NAN_EXPONENT 0x7fff

#define RJ_EXPONENT_MASK 0x7fff
#define IMPLICIT_NORMALIZED_BIT ((uint128_t)1 << EXPLICIT_MANTISSA_BITS)
/*                                 0x00010000000000000000000000000000 */
#define MANTISSA_MASK (IMPLICIT_NORMALIZED_BIT - 1)
/*                                 0x0000ffffffffffffffffffffffffffff */
#define SIGN_BIT ((uint128_t)1 << 127)
/*                                 0x80000000000000000000000000000000 */
#define MAX_EXACTLY_REPRESENTABLE_UINT128 (~(uint128_t)0 << EXPONENT_FIELD_BITS)
/*                                 0xffffffffffffffffffffffffffff8000 */

/*
 *  When |x| >= this value (0x0010000000000000), |x| == AINT(|x|) and thus
 *  has no significant digits in its fractional part.
 */
#define MIN_ENTIRELY_INTEGER IMPLICIT_NORMALIZED_BIT

#define MAX_INT_DECIMAL_DIGITS 4933 /* 1.0e4932 < max finite < 1.0e4933 */

/*
 *  The least 128-bit subnormal number > 0 needs 16494 digits in the
 *  the exact decimal representation of its fractional part to be exact.
 *  The first 4931 of them are zeroes:
 *    0.(4931 '0's)(11563 digits, ending of course in '5')
 */
#define MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS (16494 - 4931) /* 11563 */

/*
 *  Get the raw binary representation of a float128_t by means of
 *  a cast that eludes warnings from GCC -Wall.
 */
union raw_fp {
  float128_t q;
  uint128_t i;
};
#define RAW_BITS(x) (((union raw_fp *)&(x))->i)

#define SHIFT_HH32 96 /* shift value to get bit 96 to 127 of frac */
#define SHIFT_HL32 64 /* shift value to get high 64 to 95 frac */
#define SHIFT_LH32 32 /* shift value to get high 32 to 63 of frac */
#define INT_SIZE 32   /* bit size of integer type */
#define SHIFT_32 5    /* 32 is 2^5 */
#define BIT_SIZE 128  /* bit size of quad precision */
#define BYTE_SIZE 4   /* byte size of quad precision */
#define REMAIN_BIT 4  /* remain bit to multiply 10 */
#define WORDS_NUM1                                                             \
  512 /* base-(2**32) digits in little-endian order. value of 2^14/32 */
#define WORDS_NUM2                                                             \
  516 /* base 2**-32 digits in big-endian order. value of 2^14/32 + 4 */
#define INFINITY_STR_LEN 8        /* len of strings "Infinity". */
#define INF_STR_LEN 3             /* len of strings "Inf". */
#define DIGITS_BILLION 9          /* digits numbers of 1000000000 */
#define DIGITS_HUNDRED 2          /* digits numbers of 100 */
#define BASE_NUM 100              /* value of base is 100 */
#define BILLION 1000000000        /* number of billon */
#define MASK_4BIT 0xf             /* mask of 4 bits */
#define FACTOR_10 10              /* factor 10 */
#define DIGIT_5 5                 /* digit 5 */
#define DIGIT_10 10               /* digit 10 */
#define ESN_DIGITS 3              /* extra digits of ESN */
#define ESN_SCALE 3               /* scaling of ESN */
#define SUBSCRIPT_2 2             /* subscript is 2 */
#define SUBSCRIPT_3 3             /* subscript is 3 */
#define SUBSCRIPT_4 4             /* subscript is 4 */
#define DEFAULT_EXPONENT_DIGITS 2 /* default digits of exponent */
#define EXPONENT_DIGIT3 3         /* digits of exponent is 3 */
#define MAX_EXPONENT_DIGITS 4     /* max digits of exponent */
#define MAX_EXPONENT_VALUE1 9     /* max vallue of exponent while digit is 1 */
#define MAX_EXPONENT_VALUE2 99    /* max vallue of exponent while digit is 2 */
#define MAX_EXPONENT_VALUE3 999   /* max vallue of exponent while digit is 3 */
#define TRAILING_BLANK2 2         /* trailing blank is 2 */
#define TRAILING_BLANK4 4         /* trailing blank is 4 */

/*
 *  These are inline variants of memset()/memcpy() used inline for really
 *  short strings.  Note that they return the incremented output pointer.
 */

static inline char *
fill(char *out, int ch, int n)
{
  while (n-- > 0) {
    *out++ = ch;
  }
  return out;
}

static inline char *
copy(char *out, const char *str, int n)
{
  while (n-- > 0) {
    *out++ = *str++;
  }
  return out;
}

/* Format Inf and Nan */
static void
nan_or_infinite(char *out, int width, uint128_t raw, int sign_char)
{
  const char *str = NULL;
  int sign_width;

  if ((raw & MANTISSA_MASK) != 0) {
    sign_width = sign_char = 0;
    str = "NaN";
  } else {
    if (width < INFINITY_STR_LEN + (sign_char != 0))
      str = "Inf";
    else
      str = "Infinity";
    sign_width = sign_char != 0;
  }

  if (width < strlen(str) + sign_width) {
    fill(out, '*', width);
  } else {
    out = fill(out, ' ', width - (strlen(str) + sign_width));
    if (sign_char)
      *out++ = sign_char;
    copy(out, str, strlen(str));
  }
}

static char base100[BASE_NUM][2] = {
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "96", "97", "98", "99"};

/*
 *  Subroutines for format_int_part below.
 *  Format a uint128_t from right to left in a buffer.
 *  Returns a pointer to the first (most significant) digit,
 *  or NULL if overflow would have occured.  Pass a buffer
 *  pointer to the byte just after the first (least significant)
 *  digit to be written.  Assumes that the compiler can
 *  generate a fast division by a constant via multiplication
 *  by a fixed-point reciprocal.
 */
static inline char *
reversed_uint128(char *reverse_out, char *limit, uint128_t x)
{
  while (x > DIGITS_BILLION) {
    uint128_t hundredth = x / BASE_NUM;
    int rem = x - BASE_NUM * hundredth;
    if (reverse_out <= limit + 1)
      return NULL; /* overflow */
    *--reverse_out = base100[rem][1];
    *--reverse_out = base100[rem][0];
    x = hundredth;
  }

  if (x != 0) {
    if (reverse_out <= limit)
      return NULL; /* overflow */
    *--reverse_out = '0' + x;
  }
  return reverse_out;
}

/*
 *  Like reversed_uint128, but limited to [0 .. 999999999], and
 *  emits exactly 9 decimal digits, including (what will become)
 *  leading zeroes.
 */
static inline void
reversed_base_billion(char *reverse_out, char *limit, uint32_t x)
{
  int j;

  for (j = 0; j < DIGITS_BILLION - 1; j += DIGITS_HUNDRED) {
    uint32_t hundredth = x / BASE_NUM;
    int rem = x - BASE_NUM * hundredth;
    *--reverse_out = base100[rem][1];
    *--reverse_out = base100[rem][0];
    x = hundredth;
  }

  /* ninth digit (most significant) */
  *--reverse_out = '0' + x;
}

/*
 *  Divide a little-endian base-(2^32) number by one billion
 *  (in American usage, i.e. 10**9), returning the remainder and
 *  reducing the word count if possible.  Exact.  Could be generalized
 *  to accept an arbitrary divisor as an additional argument, but the
 *  PGI C compiler fails to inline and substitute.  Assumes that the
 *  compiler can generate a fast division by a constant via multiplication
 *  by a fixed-point reciprocal.
 */
static inline uint32_t
div_by_billion(uint32_t le_x[WORDS_NUM1], int *words)
{
  uint32_t remainder = 0;
  int j = *words;

  while (j-- > 0) {
    uint64_t numerator = ((uint64_t)remainder << INT_SIZE) | le_x[j];
    uint64_t quotient = numerator / BILLION;
    remainder = numerator - BILLION * quotient;
    le_x[j] = quotient;
  }

  for (j = *words; j-- > 0;) {
    if (le_x[j] != 0)
      break;
  }
  *words = j + 1;

  return remainder;
}

static inline uint128_t
quad_to_uint128(float128_t x)
{
  return (uint128_t)x;
}

/*
 *  Convert a nonnegative integer represented as a float128_t
 *  into a sequence of decimal digit characters ('0' to '9').
 *  Exact.  Returns the number of digits written.  No space fill.
 *  Nothing is written if the argument is zero.
 */
static inline int
format_int_part(char *buff, int width, float128_t x)
{
  char *out = buff + width; /* just past last character */

  /* If the integer part of x can be cast to uint64_t without overflow,
   * use a faster specialized method and avoid the multiple-word
   * arithmetic below.
   */
  if (x <= MAX_EXACTLY_REPRESENTABLE_UINT128) {
    out = reversed_uint128(out, buff, quad_to_uint128(x));
    if (out == NULL)
      return width + 1; /* overflow */
  } else {

    /* Use exact arithmetic on base-(2**32) digits represented
     * in a little-endian sequence.
     */

    uint128_t raw = RAW_BITS(x);
    uint128_t frac = (raw & MANTISSA_MASK) | IMPLICIT_NORMALIZED_BIT;
    int biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
    int unbiased_exponent = biased_exponent - EXPONENT_BIAS;
    int shift = unbiased_exponent - (EXPLICIT_MANTISSA_BITS + 1);
    uint32_t word[WORDS_NUM1]; /* base-(2**32) digits in little-endian order.
                                  2^14 / 32 */
    int words = 0;

    if (shift <= 0) {
      frac >>= -shift;
    } else {
      while (shift >= INT_SIZE) {
        word[words++] = 0;
        shift -= INT_SIZE;
      }
      if (frac != 0) {
        word[words++] = frac << shift;
        frac >>= INT_SIZE - shift;
      }
    }
    while (frac != 0) {
      word[words++] = frac;
      frac >>= INT_SIZE;
    }

    for (; words > DIGITS_HUNDRED; out -= DIGITS_BILLION) {
      uint32_t rem = div_by_billion(word, &words);
      if (out < buff + DIGITS_BILLION)
        return width + 1; /* overflow */
      reversed_base_billion(out, buff, rem);
    }

    if (words > 0) {
      uint128_t ix = word[0];
      if (words > 1)
        ix |= (uint128_t)word[1] << SHIFT_LH32;
      if (words > SUBSCRIPT_2)
        ix |= (uint128_t)word[SUBSCRIPT_2] << SHIFT_HL32;
      if (words > SUBSCRIPT_3)
        ix |= (uint128_t)word[SUBSCRIPT_3] << SHIFT_HH32;
      out = reversed_uint128(out, buff, ix);
      if (out == NULL)
        return width + 1; /* overflow */
    }
  }

  return buff + width - out;
}

/*
 *  Multiply a big-endian base-(2**-32) digit sequence in place
 *  by a uint32_t value.  Returns the carry.
 */
static inline uint32_t
mult_fraction(uint32_t be_x[WORDS_NUM2], int words, uint32_t factor)
{
  uint128_t carry = 0, factor_128 = factor;
  uint32_t *x = be_x + words;
  while (x-- > be_x) {
    carry += factor_128 * *x; /* 128-bit multiplication */
    *x = carry;
    carry >>= INT_SIZE;
  }
  return carry;
}

static inline uint32_t
mult_delimited_fraction(uint32_t be_x[WORDS_NUM2], int *words,
                        int *upper_zero_words, uint32_t factor)
{
  int up0 = *upper_zero_words, wc = *words;
  uint32_t carry = mult_fraction(be_x + up0, wc - up0, factor);
  if (up0 > 0 && carry != 0) {
    be_x[ *upper_zero_words = up0 - 1] = carry;
    carry = 0;
  }
  while (wc > 0 && be_x[wc - 1] == 0) {
    *words = --wc;
  }
  return carry;
}

/*
 *  Add '1' to a string of decimal digits; returns true
 *  when 999...9 overflows.
 */
static inline bool
decimal_increment(char *buff, int width)
{
  char *digit = buff + width;
  while (digit-- > buff) {
    if (++*digit <= '9')
      return false;
    *digit = '0';
  }
  return true; /* overflow */
}

/*
 *  Generate the digits of the decimal representation of the fractional
 *  part of a nonnegative float128_t.  Exact up to width.  Returns the next
 * digit (as int, not decimal character) and a guard flag to guide rounding.
 */
static inline void
format_fraction(char buff[MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS],
                int *next_digit_for_rounding, bool *is_inexact, int width,
                float128_t absx)
{
  uint128_t raw, frac;
  int biased_exponent, shift;
  uint32_t word[WORDS_NUM2]; /* base 2**-32 digits in big-endian order */
  int words = 0, upper_zero_words = 0;
  char *out = buff, *end = buff + width;

  *next_digit_for_rounding = 0;
  *is_inexact = false;

  if (absx >= MIN_ENTIRELY_INTEGER) {
    fill(buff, '0', width);
    return;
  }

  absx -= quad_to_uint128(absx);
  if (absx == 0.0) {
    fill(buff, '0', width);
    return;
  }

  raw = RAW_BITS(absx);
  biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
  frac = raw & MANTISSA_MASK;
  if (biased_exponent == 0) {
    /* subnormal */
    shift = EXPONENT_BIAS - (EXPONENT_FIELD_BITS + 1);
  } else {
    frac |= IMPLICIT_NORMALIZED_BIT;
    shift = EXPONENT_BIAS - biased_exponent - EXPONENT_FIELD_BITS;
  }

  if (shift < 0) {
    frac <<= -shift;
  } else {
    words = upper_zero_words = shift >> SHIFT_32;
    shift &= (INT_SIZE - 1);
    word[words++] = frac >> (shift + SHIFT_HH32);
    frac <<= INT_SIZE - shift;
  }
  while (frac != 0) {
    word[words++] = frac >> SHIFT_HH32;
    frac <<= INT_SIZE;
  }

  /* Fast-forward through any leading zeroes by stepping by billions. */
  while (upper_zero_words > 0 && out + DIGITS_BILLION <= end) {
    out = fill(out, '0', DIGITS_BILLION);
    mult_delimited_fraction(word, &words, &upper_zero_words, BILLION);
  }

  /* Extract some single digits the slow way. */
  while (words > BYTE_SIZE ||
         (words == BYTE_SIZE && ((word[BYTE_SIZE - 1] & MASK_4BIT) != 0))) {
    int digit =
        mult_delimited_fraction(word, &words, &upper_zero_words, FACTOR_10);
    if (out == end) {
      *next_digit_for_rounding = digit;
      *is_inexact = words != 0;
      return;
    }
    *out++ = '0' + digit;
  }

  /* Extract remaining single digits the fast way. */
  if (words > 0) {
    raw = upper_zero_words > 0
              ? 0
              : (uint128_t)word[0] << (SHIFT_HH32 - REMAIN_BIT);
    if (words > 1) {
      raw |= (uint128_t)word[1] << (SHIFT_HL32 - REMAIN_BIT);
      if (words > SUBSCRIPT_2) {
        raw |= (uint128_t)word[SUBSCRIPT_2] << (SHIFT_LH32 - REMAIN_BIT);
        if (words > SUBSCRIPT_3)
          raw |= word[SUBSCRIPT_3] >> REMAIN_BIT;
      }
    }
    do {
      int digit = (raw *= FACTOR_10) >> (BIT_SIZE - REMAIN_BIT);
      raw &= (~(uint128_t)0) >> REMAIN_BIT;
      if (out == end) {
        *next_digit_for_rounding = digit;
        *is_inexact = raw != 0;
        return;
      }
      *out++ = '0' + digit;
    } while (true);
  }

  fill(out, '0', end - out);
}

/*
 *  Variant of format_fraction() above that counts leading zeroes rather
 *  than emitting them.  Returns -1 if the fractional part is all zero,
 *  in which case the output field is not filled with '0' digits.
 */
static inline int
fraction_digits(char buff[MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS],
                int *next_digit_for_rounding, bool *is_inexact, int width,
                float128_t absx)
{
  uint128_t raw, frac;
  int biased_exponent, shift;
  uint32_t word[WORDS_NUM2]; /* base 2**-32 digits in big-endian order */
  int words = 0, upper_zero_words = 0;
  char *out = buff, *end = buff + width;
  int frac_leading_zeroes = 0;
  bool elide_leading_zeroes = true;

  *next_digit_for_rounding = 0;
  *is_inexact = false;

  if (absx >= MIN_ENTIRELY_INTEGER)
    return -1;
  absx -= quad_to_uint128(absx);
  if (absx == 0.0)
    return -1;

  raw = RAW_BITS(absx);
  biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
  frac = raw & MANTISSA_MASK;
  if (biased_exponent == 0) {
    /* subnormal */
    shift = EXPONENT_BIAS - (EXPONENT_FIELD_BITS + 1);
  } else {
    frac |= IMPLICIT_NORMALIZED_BIT;
    shift = EXPONENT_BIAS - biased_exponent - EXPONENT_FIELD_BITS;
  }

  if (shift < 0) {
    frac <<= -shift;
  } else {
    words = upper_zero_words = shift >> SHIFT_32;
    shift &= (INT_SIZE - 1);
    if (frac != 0) {
      word[words++] = frac >> (shift + SHIFT_HH32);
      frac <<= INT_SIZE - shift;
    }
  }

  while (frac != 0) {
    word[words++] = frac >> SHIFT_HH32;
    frac <<= INT_SIZE;
  }

  /* Fast-forward through any leading zeroes by stepping by billions. */
  while (upper_zero_words > 0) {
    frac_leading_zeroes += DIGITS_BILLION;
    mult_delimited_fraction(word, &words, &upper_zero_words, BILLION);
  }

  /* Extract single digits the slow way. */
  while (words > BYTE_SIZE ||
         (words == BYTE_SIZE && ((word[BYTE_SIZE - 1] & MASK_4BIT) != 0))) {
    int digit =
        mult_delimited_fraction(word, &words, &upper_zero_words, FACTOR_10);
    if (elide_leading_zeroes) {
      if (digit == 0) {
        ++frac_leading_zeroes;
        continue;
      }
      elide_leading_zeroes = false;
    }
    if (out == end) {
      *next_digit_for_rounding = digit;
      *is_inexact = words != 0;
      return frac_leading_zeroes;
    }
    *out++ = '0' + digit;
  }

  /* Extract remaining single digits the fast way. */
  if (words > 0) {
    raw = upper_zero_words > 0
              ? 0
              : (uint128_t)word[0] << (SHIFT_HH32 - REMAIN_BIT);
    if (words > 1) {
      raw |= (uint128_t)word[1] << (SHIFT_HL32 - REMAIN_BIT);
      if (words > SUBSCRIPT_2) {
        raw |= (uint128_t)word[SUBSCRIPT_2] << (SHIFT_LH32 - REMAIN_BIT);
        if (words > SUBSCRIPT_3)
          raw |= word[SUBSCRIPT_3] >> REMAIN_BIT;
      }
    }
    if (elide_leading_zeroes) {
      uint128_t raw10 = FACTOR_10 * raw;
      while (raw10 < ((uint128_t)1 << (BIT_SIZE - REMAIN_BIT))) {
        raw = raw10;
        raw10 *= FACTOR_10;
        ++frac_leading_zeroes;
      }
    }
    do {
      int digit = (raw *= FACTOR_10) >> (BIT_SIZE - REMAIN_BIT);
      raw &= (~(uint128_t)0) >> REMAIN_BIT;
      if (out == end) {
        *next_digit_for_rounding = digit;
        *is_inexact = raw != 0;
        return frac_leading_zeroes;
      }
      *out++ = '0' + digit;
    } while (true);
  }
  fill(out, '0', end - out);
  return frac_leading_zeroes;
}

/*
 *  Discern the CPU's current FPCR rounding mode by the portable means
 *  of observing its effect on floating-point addition.
 */
static inline enum decimal_rounding
discover_native_rounding_mode(void)
{
  static const float128_t big_pos = MAX_EXACTLY_REPRESENTABLE_UINT128;
  float128_t big_neg = -big_pos;
  if (big_pos + 1 > big_pos)
    return DECIMAL_ROUND_UP;
  if (big_neg + 1 > big_neg)
    return DECIMAL_ROUND_IN;
  if (big_neg - 1 < big_neg)
    return DECIMAL_ROUND_DOWN;
  return DECIMAL_ROUND_NEAREST;
}

/*
 *  Predicate that determines whether a decimal number
 *  should be rounded up, given the mode, next digit value,
 *  sign, and exactitude.
 */
static inline bool
should_round_up(enum decimal_rounding mode, int next_decimal_digit,
                bool is_inexact, int last_decimal_char_or_0,
                bool value_is_negative)
{
  if (next_decimal_digit == 0 && !is_inexact)
    return false; /* exact result */
  if (mode == DECIMAL_ROUND_PROCESSOR_DEFINED)
    mode = discover_native_rounding_mode();
  switch (mode) {
  case DECIMAL_ROUND_IN:
    return false;
  case DECIMAL_ROUND_UP:
    return !value_is_negative;
  case DECIMAL_ROUND_DOWN:
    return value_is_negative;
  case DECIMAL_ROUND_NEAREST:
    if (next_decimal_digit != DIGIT_5)
      return next_decimal_digit > DIGIT_5;
    /* tie: round to make even */
    if (is_inexact)
      return true;
    if (last_decimal_char_or_0 == 0)
      return false;
    return ((last_decimal_char_or_0 - '0') & 1) == 1;
  case DECIMAL_ROUND_COMPATIBLE:
  default:
    return next_decimal_digit >= DIGIT_5;
  }
}

/* Formats 0 - 9999 as decimal. */
static inline void
format_expo(char buffer[MAX_EXPONENT_DIGITS], int expo)
{
  char *out = buffer + MAX_EXPONENT_DIGITS;
  int j;
  for (j = 0; j < MAX_EXPONENT_DIGITS; ++j) {
    int tenth = expo / DIGIT_10;
    *--out = '0' + expo - DIGIT_10 * tenth;
    expo = tenth;
  }
}

/* Predicate: is a string entirely '0'? */
static inline bool
all_zeroes(const char *p, int n)
{
  while (n-- > 0) {
    if (*p++ != '0')
      return false;
  }
  return true;
}

/*
 *  Fortran Fw.d output edit descriptor with no 'kP' scaling.
 */
static inline void
F_format(char *output_buffer, int width,
         const struct formatting_control *control, float128_t x)
{
  char *out = output_buffer;
  uint128_t raw = RAW_BITS(x);
  bool is_negative = (raw & SIGN_BIT) != 0;
  float128_t absx = is_negative ? -x : x;
  int biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
  int sign_char = is_negative ? '-' : control->plus_sign;

  if (biased_exponent == INF_OR_NAN_EXPONENT) {
    /* exponent compatible with Gfortran. */
    nan_or_infinite(out, width, raw, sign_char);
  } else {
    int frac_digits = control->fraction_digits;
    int sign_width = sign_char != '\0';
    int int_part_width = width - (1 /* . */ + frac_digits);
    int int_part_digits;
    char *frac = out + int_part_width + 1;
    bool int_part_is_zero = false;

    if (int_part_width < sign_width) {
      fill(out, '*', width);
      return;
    }

    /* For SS,0P,F6.3 editing of pi:
     *   width = 6
     *   frac_digits = 3 ("141")
     *   sign_width = 0
     *   int_part_width = 2
     *   int_part_digits = 1  ("3")
     *   frac = &output_buffer[3]
     * and the '.' will be stored to output_buffer[2]
     */

    if (absx >= MIN_ENTIRELY_INTEGER) {
      /* |x| is an integer (no bits worth < 2**0) */
      fill(frac, '0', frac_digits);
    } else {
      uint128_t int_absx = quad_to_uint128(absx);
      int next_digit_for_rounding = 0;
      bool is_inexact = false;
      format_fraction(frac, &next_digit_for_rounding, &is_inexact, frac_digits,
                      absx);
      if (should_round_up(
              control->rounding, next_digit_for_rounding, is_inexact,
              frac_digits < 1 ? 0 : frac[frac_digits - 1], is_negative)) {
        if (decimal_increment(frac, frac_digits))
          ++int_absx; /* fraction rounded .999..9 up to 1.000..0 */
      }
      absx = int_absx;
    }

    int_part_is_zero = absx == 0.0;
    if (int_part_is_zero) {
      if (is_negative && control->no_minus_zero &&
          all_zeroes(frac, frac_digits)) {
        /* don't emit -.000... */
        if ((sign_char = control->plus_sign) == '\0') {
          sign_width = 0;
        }
      }
      if (!control->format_F0 && int_part_width > sign_width) {
        out[int_part_width - 1] = '0';
        int_part_digits = 1;
      } else {
        int_part_digits = frac_digits == 0;
      }
    } else {
      int_part_digits = format_int_part(out, int_part_width, absx);
    }

    if (sign_width + int_part_digits > int_part_width) {
      fill(output_buffer, '*', width);
      return;
    }

    /* Leading space fill for right justification, then sign (if any) */
    out = fill(out, ' ', int_part_width - (sign_width + int_part_digits));
    if (sign_char != '\0')
      *out++ = sign_char;

    /* Write the decimal point */
    out[int_part_digits] = control->point_char;
  }
}

/*
 *  Fortran Fw.d output edit descriptor with 'kP' scaling in effect.
 *
 *  Unlike the faster F_format() routine above, which formats directly
 *  into the output buffer, this variant formats the integer and fractional
 *  parts into a big stack buffer so that scaling can be applied more
 *  easily.
 */
static void
F_format_with_scaling(char *output_buffer, int width,
                      const struct formatting_control *control, float128_t x)
{
  char *out = output_buffer;
  uint128_t raw = RAW_BITS(x);
  bool is_negative = (raw & SIGN_BIT) != 0;
  float128_t absx = is_negative ? -x : x;
  int biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
  int sign_char = is_negative ? '-' : control->plus_sign;

  if (biased_exponent == INF_OR_NAN_EXPONENT) {
    nan_or_infinite(out, width, raw, sign_char);
  } else {

    char buffer[MAX_INT_DECIMAL_DIGITS +
                MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS];
    int int_part_digits = format_int_part(buffer, MAX_INT_DECIMAL_DIGITS, absx);
    char *payload = buffer + MAX_INT_DECIMAL_DIGITS - int_part_digits;
    int sign_width = sign_char != '\0';
    int frac_digits = control->fraction_digits;
    int scaling = control->scale_factor;
    int effective_frac_digits = frac_digits + scaling;
    int next_digit_for_rounding = 0, last_digit_for_rounding = 0;
    bool is_inexact = false;
    if (effective_frac_digits > MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS)
      effective_frac_digits = MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS;
    else if (effective_frac_digits < 0)
      effective_frac_digits = 0;
    format_fraction(payload + int_part_digits, &next_digit_for_rounding,
                    &is_inexact, effective_frac_digits, absx);

    while (scaling > 0 && effective_frac_digits > 0) {
      if (int_part_digits == 0 && *payload == '0') {
        ++payload;
      } else {
        ++int_part_digits;
      }
      --scaling;
      --effective_frac_digits;
    }
    while (scaling < 0) {
      if (int_part_digits > 0) {
        --int_part_digits;
      } else {
        *--payload = '0';
      }
      ++effective_frac_digits;
      ++scaling;
    }

    while (effective_frac_digits > frac_digits) {
      if (next_digit_for_rounding != 0)
        is_inexact = true;
      next_digit_for_rounding =
          payload[int_part_digits + --effective_frac_digits] - '0';
    }
    if (int_part_digits + frac_digits > 0)
      last_digit_for_rounding = payload[int_part_digits + frac_digits - 1];
    if (should_round_up(control->rounding, next_digit_for_rounding, is_inexact,
                        last_digit_for_rounding, is_negative)) {
      if (decimal_increment(payload, int_part_digits + frac_digits)) {
        *--payload = '1';
        ++int_part_digits;
      }
    }

    if (is_negative && control->no_minus_zero &&
        all_zeroes(payload, int_part_digits + frac_digits)) {
      /* don't emit -.000... */
      if ((sign_char = control->plus_sign) == '\0') {
        sign_width = 0;
      }
    }

    if (!control->format_F0 && int_part_digits == 0 &&
        (frac_digits == 0 || sign_width + 1 /* . */ + frac_digits < width)) {
      *--payload = '0';
      int_part_digits = 1;
    }

    if (sign_width + int_part_digits + 1 /* . */ + frac_digits > width) {
      fill(out, '*', width);
    } else {
      out = fill(out, ' ',
                 width - (sign_width + int_part_digits + 1 + frac_digits));
      if (sign_char != '\0')
        *out++ = sign_char;
      out = copy(out, payload, int_part_digits);
      *out++ = control->point_char;
      copy(out, payload + int_part_digits, frac_digits);
    }
  }
}

/*
 *  Fortran Ew.d, Ew.dEe, Dw.d, ESw.d, and ENw.d output edit descriptors
 */
static inline void
ED_format(char *out_buffer, int width, const struct formatting_control *control,
          int E_or_D_char, float128_t x)
{
  char *out = out_buffer;
  uint128_t raw = RAW_BITS(x);
  bool is_negative = (raw & SIGN_BIT) != 0;
  float128_t absx = is_negative ? -x : x;
  int biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
  int sign_char = is_negative ? '-' : control->plus_sign;

  if (biased_exponent == INF_OR_NAN_EXPONENT) {
    if (control->format_G0)
      width = INF_STR_LEN + (sign_char != 0);
    nan_or_infinite(out, width, raw, sign_char);
  } else {

    /* Note that, per the Fortran standard:
     *  - "ES" and "EN" formatting take effect *before* decimal rounding.
     *  - So does "kP" scaling, in contradistinction to "Fw.d" formatting
     *    (above), which must apply scaling *after* rounding.  Thus "ES"
     *    formatting seems to be identical to "1P" scaling with "Ew.d",
     *    although the Fortran 2008 standard never actually states so.
     *  - The "kP" scaling factor is ignored with "ES"/"EN" formatting.
     *  - "ES" formatting produces 1 extra significant digit.
     *  - Scaling factors kP with k > 0 produce 1 extra significant digit.
     *  - "EN" formatting produces 1 to 3 extra significant digits.
     */

    char buffer[MAX_INT_DECIMAL_DIGITS +
                MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS];
    int int_part_digits = format_int_part(buffer, MAX_INT_DECIMAL_DIGITS, absx);
    char *payload = buffer + MAX_INT_DECIMAL_DIGITS - int_part_digits;
    int trailing_zeroes = 0;
    int ESN = control->ESN_format;
    int extra_digits = ESN == 'N'                  ? ESN_DIGITS
                       : ESN == 'S'                ? 1
                       : control->format_G0 == 1   ? 0
                       : control->scale_factor > 0 ? 1
                                                   : 0;
    int scaling = ESN == 'N'   ? ESN_SCALE
                  : ESN == 'S' ? 1
                               : control->scale_factor;
    int lost_digits = scaling < 0 ? -scaling : 0;
    int sign_width = sign_char != '\0';
    int ED_char_width = 1;
    int frac_digits = control->fraction_digits;
    int explicit_expo_digits = control->exponent_digits;
    int expo_digits = explicit_expo_digits;
    int significant_digits = frac_digits + extra_digits - lost_digits;
    int frac_part_digits = significant_digits - int_part_digits;
    int expo, abs_expo;
    int leading_spaces;
    bool all_digits_zero = false;
    int next_digit_for_rounding = 0, last_digit_for_rounding = 0;
    bool is_inexact = false;

    if (frac_part_digits > MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS) {
      trailing_zeroes =
          frac_part_digits - MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS;
      frac_part_digits = MAX_FRACTION_SIGNIFICANT_DECIMAL_DIGITS;
    }

    if (int_part_digits == 0) {
      int frac_leading_zeroes =
          fraction_digits(payload, &next_digit_for_rounding, &is_inexact,
                          frac_part_digits, absx);
      all_digits_zero = frac_leading_zeroes < 0;
      expo = all_digits_zero ? 0 : -frac_leading_zeroes;
    } else if (frac_part_digits < 0) {
      expo = int_part_digits;
      is_inexact = absx < MAX_EXACTLY_REPRESENTABLE_UINT128 &&
                   absx != quad_to_uint128(absx);
      while (int_part_digits > significant_digits) {
        is_inexact |= next_digit_for_rounding != 0;
        next_digit_for_rounding = payload[--int_part_digits] - '0';
      }
    } else {
      format_fraction(payload + int_part_digits, &next_digit_for_rounding,
                      &is_inexact, frac_part_digits, absx);
      expo = int_part_digits;
    }

    /* "Engineering" (EN) format: ensure that the exponent is a multiple of 3.
     * extra_digits and scaling are both 3 now, but they can be reduced
     * to 2 or 1 in order to make the exponent a multiple of three.
     */
    if (ESN == 'N') {
      int int_digits;
      if (expo <= 0)
        int_digits = ESN_DIGITS - (-expo % ESN_DIGITS);
      else
        int_digits = ((expo - 1) % ESN_DIGITS) + 1;
      while (int_digits++ < ESN_DIGITS) {
        if (trailing_zeroes > 0) {
          --trailing_zeroes;
        } else {
          is_inexact |= next_digit_for_rounding != 0;
          next_digit_for_rounding = payload[frac_digits + --extra_digits] - '0';
        }
      }
      scaling = extra_digits;
      significant_digits = frac_digits + extra_digits; /* lost_digits == 0 */
    }

    /*
     *  Decimal rounding.  (Brent observes that the last significant decimal
     *  digit in an exact conversion of a nonzero fractional part must be 5,
     *  but I don't see a way to exploit that cool fact here other than
     *  by noting it for posterity.)
     */
    if (!all_digits_zero && significant_digits >= 1)
      last_digit_for_rounding = payload[significant_digits - 1];
    if (trailing_zeroes == 0 &&
        should_round_up(control->rounding, next_digit_for_rounding, is_inexact,
                        last_digit_for_rounding, is_negative)) {
      if (decimal_increment(payload, significant_digits)) {
        *--payload = '1';
        ++expo;
        if (ESN == 'N') {
          if (++extra_digits > ESN_DIGITS) {
            /* 999.99... -> 1.00000... */
            extra_digits -= ESN_DIGITS;
          }
          scaling = extra_digits;
        }
      }
      all_digits_zero = false;
    }

    if (expo_digits == 0)
      expo_digits =
          DEFAULT_EXPONENT_DIGITS; /* can grow to 4 below if expo>999! */

    if (all_digits_zero) {
      /* There are just too many special cases that apply to zero
       * values, so format them here directly in order to simplify
       * the general code path below.
       */
      if (extra_digits > 1)
        extra_digits = 1;
      if (is_negative && control->no_minus_zero) {
        /* option: don't emit -0.000... */
        if ((sign_char = control->plus_sign) == '\0')
          sign_width = 0;
      }
      leading_spaces =
          width - (sign_width + extra_digits + 1 /* . */ + frac_digits +
                   ED_char_width + 1 /* + */ + expo_digits);
      if (leading_spaces < 0) {
        fill(out, '*', width);
      } else {
        if (extra_digits == 0 && leading_spaces > 0) {
          extra_digits = 1;
          --leading_spaces;
        }
        out = fill(out, ' ', leading_spaces);
        if (sign_char != '\0')
          *out++ = sign_char;
        if (extra_digits > 0)
          *out++ = '0';
        *out++ = control->point_char;
        out = fill(out, '0', frac_digits);
        *out++ = E_or_D_char;
        *out++ = '+';
        fill(out, '0', expo_digits);
      }
      return;
    }

    expo -= scaling;
    abs_expo = expo >= 0 ? expo : -expo;
    if (abs_expo > MAX_EXPONENT_VALUE2 && abs_expo <= MAX_EXPONENT_VALUE3 &&
        expo_digits == DEFAULT_EXPONENT_DIGITS && explicit_expo_digits == 0) {
      /* No explicit "Ee" exponent width was supplied, and the
       * actual exponent won't fit in 2 digits (E+nn); use +nnn.
       */
      if (!control->format_G0)
        ED_char_width = 0;
      expo_digits = DEFAULT_EXPONENT_DIGITS + 1;
    } else if (abs_expo > MAX_EXPONENT_VALUE3 &&
               (control->format_G0 ||
                /* deal with FED_E, FED_D and FED_G compatible with Gfortran. */
                expo_digits == DEFAULT_EXPONENT_DIGITS + 1)) {
      /* No explicit "Ee" exponent width was supplied, and the
       * actual exponent won't fit in 3 digits (E+nnn); use +nnnn.
       */
      if (!control->format_G0)
        ED_char_width = 0;
      expo_digits = MAX_EXPONENT_DIGITS;
    }

    if ((abs_expo > MAX_EXPONENT_VALUE1 &&
         expo_digits < DEFAULT_EXPONENT_DIGITS) ||
        (abs_expo > MAX_EXPONENT_VALUE2 &&
         expo_digits < DEFAULT_EXPONENT_DIGITS + 1) ||
        (abs_expo > MAX_EXPONENT_VALUE3 && expo_digits < MAX_EXPONENT_DIGITS)) {
      fill(out, '*', width);
      return;
    }

    leading_spaces =
        width - (sign_width + extra_digits + 1 /* . */ + frac_digits +
                 ED_char_width + 1 /* + */ + expo_digits);

    if (leading_spaces < 0) {
      fill(out, '*', width);
    } else {
      char expo_buffer[MAX_EXPONENT_DIGITS];
      bool initial_zero = leading_spaces > 0 && scaling <= 0;

      if (initial_zero)
        --leading_spaces;
      out = fill(out, ' ', leading_spaces);

      if (sign_char != '\0')
        *out++ = sign_char;

      if (scaling > 0) {
        out = copy(out, payload, scaling);
        *out++ = control->point_char;
        out =
            copy(out, payload + scaling, frac_digits + extra_digits - scaling);
      } else {
        if (initial_zero)
          *out++ = '0';
        *out++ = control->point_char;
        out = fill(out, '0', -scaling);
        out = copy(out, payload, frac_digits + scaling);
      }

      out = fill(out, '0', trailing_zeroes);

      if (ED_char_width > 0)
        *out++ = E_or_D_char;
      *out++ = expo < 0 ? '-' : '+';
      if (expo_digits > MAX_EXPONENT_DIGITS) {
        out = fill(out, '0', expo_digits - MAX_EXPONENT_DIGITS);
        expo_digits = MAX_EXPONENT_DIGITS;
      }
      format_expo(expo_buffer, abs_expo);
      if (expo_digits >= MAX_EXPONENT_DIGITS)
        *out++ = expo_buffer[0];
      if (expo_digits >= EXPONENT_DIGIT3)
        *out++ = expo_buffer[1];
      if (expo_digits >= DEFAULT_EXPONENT_DIGITS)
        *out++ = expo_buffer[SUBSCRIPT_2];
      *out++ = expo_buffer[SUBSCRIPT_3];
    }
  }
}

/* Fortran Gw.d and Gw.dEe output edit descriptors */
static inline void
G_format(char *out, int width, const struct formatting_control *control,
         float128_t x)
{
  uint128_t raw = RAW_BITS(x);
  bool is_negative = (raw & SIGN_BIT) != 0;
  float128_t absx = is_negative ? -x : x;
  int biased_exponent = (raw >> EXPLICIT_MANTISSA_BITS) & RJ_EXPONENT_MASK;
  int sign_char = is_negative ? '-' : control->plus_sign;

  if (biased_exponent != INF_OR_NAN_EXPONENT) {
    int significant_digits = control->fraction_digits; /* 'd' */
    int sign_width = sign_char != '\0';
    int trailing_blanks = control->exponent_digits == 0
                              ? TRAILING_BLANK4
                              : TRAILING_BLANK2 + control->exponent_digits;
    int max_int_part_width = width - (sign_width + 1 /* . */ + trailing_blanks);
    int int_part_digits = format_int_part(out, width, absx);
    if (int_part_digits <= max_int_part_width &&
        int_part_digits <= significant_digits) {
      int frac_digits = significant_digits - int_part_digits;
      int leading_spaces = width - (sign_width + int_part_digits + 1 /* . */ +
                                    frac_digits + trailing_blanks);
      char *int_part = out + leading_spaces + sign_width;
      char *frac_part = int_part + int_part_digits + 1 /* . */;
      int next_digit_for_rounding = 0;
      bool is_inexact = false;

      if (leading_spaces < 0)
        goto do_E_formatting;

      memmove(int_part, out + width - int_part_digits,
              int_part_digits); /* left-justify the int part */
      format_fraction(frac_part, &next_digit_for_rounding, &is_inexact,
                      frac_digits, absx);

      if (should_round_up(
              control->rounding, next_digit_for_rounding, is_inexact,
              frac_digits < 1 ? 0 : frac_part[frac_digits - 1], is_negative)) {
        if (decimal_increment(frac_part, frac_digits)) {
          /* fraction rounded .999..99 up to 1.000..0 */
          if (decimal_increment(int_part, int_part_digits)) {
            /* integer part grew from 999..99. to 1000..00. */
            if (frac_digits-- <= 0)
              goto do_E_formatting;
            int_part[int_part_digits++] = '0';
            *int_part = '1';
          }
        }
      }

      if (int_part_digits == 0) {
        bool all_frac_zeroes = all_zeroes(frac_part, frac_digits);
        if ((frac_digits > 0 && frac_part[0] == '0' &&
             (!all_frac_zeroes || next_digit_for_rounding > 0 || is_inexact)) ||
            (frac_digits == 0 && control->scale_factor))
          goto do_E_formatting; /* 0 < |x| < 0.1 */
        if (all_frac_zeroes) {
          if (is_negative && control->no_minus_zero) {
            if ((sign_char = control->plus_sign) == '\0')
              sign_width = 0;
          }
          /* Use an integer part of '0' and count it as a significant digit. */
          int_part_digits = 1; /* '0' */
          if (control->fraction_digits != 0) {
            frac_digits = significant_digits - int_part_digits;
          } else {
            frac_digits = control->fraction_digits;
          }
          leading_spaces = width - (sign_width + int_part_digits + 1 +
                                    frac_digits + trailing_blanks);
          int_part = out + leading_spaces + sign_width;
          frac_part = int_part + int_part_digits + 1 /* . */;
          *int_part = '0';
          fill(frac_part, '0', frac_digits);
        } else if (leading_spaces > 0) {
          /* Emit an optional leading 0 integer part, when space allows */
          --leading_spaces;
          int_part_digits = 1;
          *--int_part = '0';
        }
      }

      memset(out, ' ', leading_spaces);
      if (sign_char != '\0')
        out[leading_spaces] = sign_char;
      int_part[int_part_digits] = control->point_char;
      memset(out + width - trailing_blanks, ' ', trailing_blanks);
      return;
    }
  }

do_E_formatting:
  /* When G fails, treat it as E. */
  ED_format(out, width, control, 'E', x);
}

/* Entry point */
void
__fortio_format_quad(char *out, int width,
                     const struct formatting_control *control, float128_t x)
{
  switch (control->format_char) {
  case 'E':
  case 'D':
    ED_format(out, width, control, control->format_char, x);
    break;
  case 'F':
    if (control->scale_factor != 0)
      F_format_with_scaling(out, width, control, x);
    else
      F_format(out, width, control, x);
    break;
  case 'G':
  default:
    G_format(out, width, control, x);
  }
}
