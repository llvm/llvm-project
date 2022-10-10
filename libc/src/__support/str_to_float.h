//===-- String to float conversion utils ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_SUPPORT_STR_TO_FLOAT_H
#define LIBC_SRC_SUPPORT_STR_TO_FLOAT_H

#include "src/__support/CPP/limits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/UInt128.h"
#include "src/__support/builtin_wrappers.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/detailed_powers_of_ten.h"
#include "src/__support/high_precision_decimal.h"
#include "src/__support/str_to_integer.h"
#include <errno.h>

namespace __llvm_libc {
namespace internal {

template <class T> uint32_t inline leading_zeroes(T inputNumber) {
  constexpr uint32_t BITS_IN_T = sizeof(T) * 8;
  if (inputNumber == 0) {
    return BITS_IN_T;
  }
  uint32_t cur_guess = BITS_IN_T / 2;
  uint32_t range_size = BITS_IN_T / 2;
  // while either shifting by curGuess does not get rid of all of the bits or
  // shifting by one less also gets rid of all of the bits then we have not
  // found the first bit.
  while (((inputNumber >> cur_guess) > 0) ||
         ((inputNumber >> (cur_guess - 1)) == 0)) {
    // Binary search for the first set bit
    range_size /= 2;
    if (range_size == 0) {
      break;
    }
    if ((inputNumber >> cur_guess) > 0) {
      cur_guess += range_size;
    } else {
      cur_guess -= range_size;
    }
  }
  if (inputNumber >> cur_guess > 0) {
    cur_guess++;
  }
  return BITS_IN_T - cur_guess;
}

template <> uint32_t inline leading_zeroes<uint32_t>(uint32_t inputNumber) {
  return safe_clz(inputNumber);
}

template <> uint32_t inline leading_zeroes<uint64_t>(uint64_t inputNumber) {
  return safe_clz(inputNumber);
}

static inline uint64_t low64(const UInt128 &num) {
  return static_cast<uint64_t>(num & 0xffffffffffffffff);
}

static inline uint64_t high64(const UInt128 &num) {
  return static_cast<uint64_t>(num >> 64);
}

template <class T> inline void set_implicit_bit(fputil::FPBits<T> &) { return; }

#if defined(SPECIAL_X86_LONG_DOUBLE)
template <>
inline void set_implicit_bit<long double>(fputil::FPBits<long double> &result) {
  result.set_implicit_bit(result.get_unbiased_exponent() != 0);
}
#endif

// This Eisel-Lemire implementation is based on the algorithm described in the
// paper Number Parsing at a Gigabyte per Second, Software: Practice and
// Experience 51 (8), 2021 (https://arxiv.org/abs/2101.11408), as well as the
// description by Nigel Tao
// (https://nigeltao.github.io/blog/2020/eisel-lemire.html) and the golang
// implementation, also by Nigel Tao
// (https://github.com/golang/go/blob/release-branch.go1.16/src/strconv/eisel_lemire.go#L25)
// for some optimizations as well as handling 32 bit floats.
template <class T>
static inline bool
eisel_lemire(typename fputil::FPBits<T>::UIntType mantissa, int32_t exp10,
             typename fputil::FPBits<T>::UIntType *outputMantissa,
             uint32_t *outputExp2) {

  using BitsType = typename fputil::FPBits<T>::UIntType;
  constexpr uint32_t BITS_IN_MANTISSA = sizeof(mantissa) * 8;

  if (sizeof(T) > 8) { // This algorithm cannot handle anything longer than a
                       // double, so we skip straight to the fallback.
    return false;
  }

  // Exp10 Range
  if (exp10 < DETAILED_POWERS_OF_TEN_MIN_EXP_10 ||
      exp10 > DETAILED_POWERS_OF_TEN_MAX_EXP_10) {
    return false;
  }

  // Normalization
  uint32_t clz = leading_zeroes<BitsType>(mantissa);
  mantissa <<= clz;

  uint32_t exp2 = static_cast<uint32_t>(exp10_to_exp2(exp10)) + BITS_IN_MANTISSA +
                  fputil::FloatProperties<T>::EXPONENT_BIAS - clz;

  // Multiplication
  const uint64_t *power_of_ten =
      DETAILED_POWERS_OF_TEN[exp10 - DETAILED_POWERS_OF_TEN_MIN_EXP_10];

  UInt128 first_approx =
      static_cast<UInt128>(mantissa) * static_cast<UInt128>(power_of_ten[1]);

  // Wider Approximation
  UInt128 final_approx;
  // The halfway constant is used to check if the bits that will be shifted away
  // intially are all 1. For doubles this is 64 (bitstype size) - 52 (final
  // mantissa size) - 3 (we shift away the last two bits separately for
  // accuracy, and the most significant bit is ignored.) = 9 bits. Similarly,
  // it's 6 bits for floats in this case.
  const uint64_t halfway_constant =
      (uint64_t(1) << (BITS_IN_MANTISSA -
                       fputil::FloatProperties<T>::MANTISSA_WIDTH - 3)) -
      1;
  if ((high64(first_approx) & halfway_constant) == halfway_constant &&
      low64(first_approx) + mantissa < mantissa) {
    UInt128 low_bits =
        static_cast<UInt128>(mantissa) * static_cast<UInt128>(power_of_ten[0]);
    UInt128 second_approx =
        first_approx + static_cast<UInt128>(high64(low_bits));

    if ((high64(second_approx) & halfway_constant) == halfway_constant &&
        low64(second_approx) + 1 == 0 &&
        low64(low_bits) + mantissa < mantissa) {
      return false;
    }
    final_approx = second_approx;
  } else {
    final_approx = first_approx;
  }

  // Shifting to 54 bits for doubles and 25 bits for floats
  BitsType msb = static_cast<BitsType>(high64(final_approx) >> (BITS_IN_MANTISSA - 1));
  BitsType final_mantissa = static_cast<BitsType>(high64(final_approx) >>
                            (msb + BITS_IN_MANTISSA -
                             (fputil::FloatProperties<T>::MANTISSA_WIDTH + 3)));
  exp2 -= static_cast<uint32_t>(1 ^ msb); // same as !msb

  // Half-way ambiguity
  if (low64(final_approx) == 0 &&
      (high64(final_approx) & halfway_constant) == 0 &&
      (final_mantissa & 3) == 1) {
    return false;
  }

  // From 54 to 53 bits for doubles and 25 to 24 bits for floats
  final_mantissa += final_mantissa & 1;
  final_mantissa >>= 1;
  if ((final_mantissa >> (fputil::FloatProperties<T>::MANTISSA_WIDTH + 1)) >
      0) {
    final_mantissa >>= 1;
    ++exp2;
  }

  // The if block is equivalent to (but has fewer branches than):
  //   if exp2 <= 0 || exp2 >= 0x7FF { etc }
  if (exp2 - 1 >= (1 << fputil::FloatProperties<T>::EXPONENT_WIDTH) - 2) {
    return false;
  }

  *outputMantissa = final_mantissa;
  *outputExp2 = exp2;
  return true;
}

#if !defined(LONG_DOUBLE_IS_DOUBLE)
template <>
inline bool eisel_lemire<long double>(
    typename fputil::FPBits<long double>::UIntType mantissa, int32_t exp10,
    typename fputil::FPBits<long double>::UIntType *outputMantissa,
    uint32_t *outputExp2) {
  using BitsType = typename fputil::FPBits<long double>::UIntType;
  constexpr uint32_t BITS_IN_MANTISSA = sizeof(mantissa) * 8;

  // Exp10 Range
  // This doesn't reach very far into the range for long doubles, since it's
  // sized for doubles and their 11 exponent bits, and not for long doubles and
  // their 15 exponent bits (max exponent of ~300 for double vs ~5000 for long
  // double). This is a known tradeoff, and was made because a proper long
  // double table would be approximately 16 times larger. This would have
  // significant memory and storage costs all the time to speed up a relatively
  // uncommon path. In addition the exp10_to_exp2 function only approximates
  // multiplying by log(10)/log(2), and that approximation may not be accurate
  // out to the full long double range.
  if (exp10 < DETAILED_POWERS_OF_TEN_MIN_EXP_10 ||
      exp10 > DETAILED_POWERS_OF_TEN_MAX_EXP_10) {
    return false;
  }

  // Normalization
  uint32_t clz = leading_zeroes<BitsType>(mantissa);
  mantissa <<= clz;

  uint32_t exp2 = static_cast<uint32_t>(exp10_to_exp2(exp10)) + BITS_IN_MANTISSA +
                  fputil::FloatProperties<long double>::EXPONENT_BIAS - clz;

  // Multiplication
  const uint64_t *power_of_ten =
      DETAILED_POWERS_OF_TEN[exp10 - DETAILED_POWERS_OF_TEN_MIN_EXP_10];

  // Since the input mantissa is more than 64 bits, we have to multiply with the
  // full 128 bits of the power of ten to get an approximation with the same
  // number of significant bits. This means that we only get the one
  // approximation, and that approximation is 256 bits long.
  UInt128 approx_upper = static_cast<UInt128>(high64(mantissa)) *
                         static_cast<UInt128>(power_of_ten[1]);

  UInt128 approx_middle = static_cast<UInt128>(high64(mantissa)) *
                              static_cast<UInt128>(power_of_ten[0]) +
                          static_cast<UInt128>(low64(mantissa)) *
                              static_cast<UInt128>(power_of_ten[1]);

  UInt128 approx_lower = static_cast<UInt128>(low64(mantissa)) *
                         static_cast<UInt128>(power_of_ten[0]);

  UInt128 final_approx_lower =
      approx_lower + (static_cast<UInt128>(low64(approx_middle)) << 64);
  UInt128 final_approx_upper = approx_upper + high64(approx_middle) +
                               (final_approx_lower < approx_lower ? 1 : 0);

  // The halfway constant is used to check if the bits that will be shifted away
  // intially are all 1. For 80 bit floats this is 128 (bitstype size) - 64
  // (final mantissa size) - 3 (we shift away the last two bits separately for
  // accuracy, and the most significant bit is ignored.) = 61 bits. Similarly,
  // it's 12 bits for 128 bit floats in this case.
  constexpr UInt128 HALFWAY_CONSTANT =
      (UInt128(1) << (BITS_IN_MANTISSA -
                      fputil::FloatProperties<long double>::MANTISSA_WIDTH -
                      3)) -
      1;

  if ((final_approx_upper & HALFWAY_CONSTANT) == HALFWAY_CONSTANT &&
      final_approx_lower + mantissa < mantissa) {
    return false;
  }

  // Shifting to 65 bits for 80 bit floats and 113 bits for 128 bit floats
  BitsType msb = final_approx_upper >> (BITS_IN_MANTISSA - 1);
  BitsType final_mantissa =
      final_approx_upper >>
      (msb + BITS_IN_MANTISSA -
       (fputil::FloatProperties<long double>::MANTISSA_WIDTH + 3));
  exp2 -= static_cast<uint32_t>(1 ^ msb); // same as !msb

  // Half-way ambiguity
  if (final_approx_lower == 0 && (final_approx_upper & HALFWAY_CONSTANT) == 0 &&
      (final_mantissa & 3) == 1) {
    return false;
  }

  // From 65 to 64 bits for 80 bit floats and 113  to 112 bits for 128 bit
  // floats
  final_mantissa += final_mantissa & 1;
  final_mantissa >>= 1;
  if ((final_mantissa >>
       (fputil::FloatProperties<long double>::MANTISSA_WIDTH + 1)) > 0) {
    final_mantissa >>= 1;
    ++exp2;
  }

  // The if block is equivalent to (but has fewer branches than):
  //   if exp2 <= 0 || exp2 >= MANTISSA_MAX { etc }
  if (exp2 - 1 >=
      (1 << fputil::FloatProperties<long double>::EXPONENT_WIDTH) - 2) {
    return false;
  }

  *outputMantissa = final_mantissa;
  *outputExp2 = exp2;
  return true;
}
#endif

// The nth item in POWERS_OF_TWO represents the greatest power of two less than
// 10^n. This tells us how much we can safely shift without overshooting.
constexpr uint8_t POWERS_OF_TWO[19] = {
    0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56, 59,
};
constexpr int32_t NUM_POWERS_OF_TWO =
    sizeof(POWERS_OF_TWO) / sizeof(POWERS_OF_TWO[0]);

// Takes a mantissa and base 10 exponent and converts it into its closest
// floating point type T equivalent. This is the fallback algorithm used when
// the Eisel-Lemire algorithm fails, it's slower but more accurate. It's based
// on the Simple Decimal Conversion algorithm by Nigel Tao, described at this
// link: https://nigeltao.github.io/blog/2020/parse-number-f64-simple.html
template <class T>
static inline void
simple_decimal_conversion(const char *__restrict numStart,
                          typename fputil::FPBits<T>::UIntType *outputMantissa,
                          uint32_t *outputExp2) {

  int32_t exp2 = 0;
  HighPrecisionDecimal hpd = HighPrecisionDecimal(numStart);

  if (hpd.get_num_digits() == 0) {
    *outputMantissa = 0;
    *outputExp2 = 0;
    return;
  }

  // If the exponent is too large and can't be represented in this size of
  // float, return inf.
  if (hpd.get_decimal_point() > 0 &&
      exp10_to_exp2(hpd.get_decimal_point() - 1) >
          static_cast<int64_t>(fputil::FloatProperties<T>::EXPONENT_BIAS)) {
    *outputMantissa = 0;
    *outputExp2 = fputil::FPBits<T>::MAX_EXPONENT;
    errno = ERANGE;
    return;
  }
  // If the exponent is too small even for a subnormal, return 0.
  if (hpd.get_decimal_point() < 0 &&
      exp10_to_exp2(-hpd.get_decimal_point()) >
          static_cast<int64_t>(fputil::FloatProperties<T>::EXPONENT_BIAS +
                               fputil::FloatProperties<T>::MANTISSA_WIDTH)) {
    *outputMantissa = 0;
    *outputExp2 = 0;
    errno = ERANGE;
    return;
  }

  // Right shift until the number is smaller than 1.
  while (hpd.get_decimal_point() > 0) {
    int32_t shift_amount = 0;
    if (hpd.get_decimal_point() >= NUM_POWERS_OF_TWO) {
      shift_amount = 60;
    } else {
      shift_amount = POWERS_OF_TWO[hpd.get_decimal_point()];
    }
    exp2 += shift_amount;
    hpd.shift(-shift_amount);
  }

  // Left shift until the number is between 1/2 and 1
  while (hpd.get_decimal_point() < 0 ||
         (hpd.get_decimal_point() == 0 && hpd.get_digits()[0] < 5)) {
    int32_t shift_amount = 0;

    if (-hpd.get_decimal_point() >= NUM_POWERS_OF_TWO) {
      shift_amount = 60;
    } else if (hpd.get_decimal_point() != 0) {
      shift_amount = POWERS_OF_TWO[-hpd.get_decimal_point()];
    } else { // This handles the case of the number being between .1 and .5
      shift_amount = 1;
    }
    exp2 -= shift_amount;
    hpd.shift(shift_amount);
  }

  // Left shift once so that the number is between 1 and 2
  --exp2;
  hpd.shift(1);

  // Get the biased exponent
  exp2 += fputil::FloatProperties<T>::EXPONENT_BIAS;

  // Handle the exponent being too large (and return inf).
  if (exp2 >= fputil::FPBits<T>::MAX_EXPONENT) {
    *outputMantissa = 0;
    *outputExp2 = fputil::FPBits<T>::MAX_EXPONENT;
    errno = ERANGE;
    return;
  }

  // Shift left to fill the mantissa
  hpd.shift(fputil::FloatProperties<T>::MANTISSA_WIDTH);
  typename fputil::FPBits<T>::UIntType final_mantissa =
      hpd.round_to_integer_type<typename fputil::FPBits<T>::UIntType>();

  // Handle subnormals
  if (exp2 <= 0) {
    // Shift right until there is a valid exponent
    while (exp2 < 0) {
      hpd.shift(-1);
      ++exp2;
    }
    // Shift right one more time to compensate for the left shift to get it
    // between 1 and 2.
    hpd.shift(-1);
    final_mantissa =
        hpd.round_to_integer_type<typename fputil::FPBits<T>::UIntType>();

    // Check if by shifting right we've caused this to round to a normal number.
    if ((final_mantissa >> fputil::FloatProperties<T>::MANTISSA_WIDTH) != 0) {
      ++exp2;
    }
  }

  // Check if rounding added a bit, and shift down if that's the case.
  if (final_mantissa == typename fputil::FPBits<T>::UIntType(2)
                            << fputil::FloatProperties<T>::MANTISSA_WIDTH) {
    final_mantissa >>= 1;
    ++exp2;

    // Check if this rounding causes exp2 to go out of range and make the result
    // INF. If this is the case, then finalMantissa and exp2 are already the
    // correct values for an INF result.
    if (exp2 >= fputil::FPBits<T>::MAX_EXPONENT) {
      errno = ERANGE; // NOLINT
    }
  }

  if (exp2 == 0) {
    errno = ERANGE;
  }

  *outputMantissa = final_mantissa;
  *outputExp2 = exp2;
}

// This class is used for templating the constants for Clinger's Fast Path,
// described as a method of approximation in
// Clinger WD. How to Read Floating Point Numbers Accurately. SIGPLAN Not 1990
// Jun;25(6):92–101. https://doi.org/10.1145/93548.93557.
// As well as the additions by Gay that extend the useful range by the number of
// exact digits stored by the float type, described in
// Gay DM, Correctly rounded binary-decimal and decimal-binary conversions;
// 1990. AT&T Bell Laboratories Numerical Analysis Manuscript 90-10.
template <class T> class ClingerConsts;

template <> class ClingerConsts<float> {
public:
  static constexpr float POWERS_OF_TEN_ARRAY[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5,
                                                  1e6, 1e7, 1e8, 1e9, 1e10};
  static constexpr int32_t EXACT_POWERS_OF_TEN = 10;
  static constexpr int32_t DIGITS_IN_MANTISSA = 7;
  static constexpr float MAX_EXACT_INT = 16777215.0;
};

template <> class ClingerConsts<double> {
public:
  static constexpr double POWERS_OF_TEN_ARRAY[] = {
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
      1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
  static constexpr int32_t EXACT_POWERS_OF_TEN = 22;
  static constexpr int32_t DIGITS_IN_MANTISSA = 15;
  static constexpr double MAX_EXACT_INT = 9007199254740991.0;
};

#if defined(LONG_DOUBLE_IS_DOUBLE)
template <> class ClingerConsts<long double> {
public:
  static constexpr long double POWERS_OF_TEN_ARRAY[] = {
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
      1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
  static constexpr int32_t EXACT_POWERS_OF_TEN =
      ClingerConsts<double>::EXACT_POWERS_OF_TEN;
  static constexpr int32_t DIGITS_IN_MANTISSA =
      ClingerConsts<double>::DIGITS_IN_MANTISSA;
  static constexpr long double MAX_EXACT_INT =
      ClingerConsts<double>::MAX_EXACT_INT;
};
#elif defined(SPECIAL_X86_LONG_DOUBLE)
template <> class ClingerConsts<long double> {
public:
  static constexpr long double POWERS_OF_TEN_ARRAY[] = {
      1e0L,  1e1L,  1e2L,  1e3L,  1e4L,  1e5L,  1e6L,  1e7L,  1e8L,  1e9L,
      1e10L, 1e11L, 1e12L, 1e13L, 1e14L, 1e15L, 1e16L, 1e17L, 1e18L, 1e19L,
      1e20L, 1e21L, 1e22L, 1e23L, 1e24L, 1e25L, 1e26L, 1e27L};
  static constexpr int32_t EXACT_POWERS_OF_TEN = 27;
  static constexpr int32_t DIGITS_IN_MANTISSA = 21;
  static constexpr long double MAX_EXACT_INT = 18446744073709551615.0L;
};
#else
template <> class ClingerConsts<long double> {
public:
  static constexpr long double POWERS_OF_TEN_ARRAY[] = {
      1e0L,  1e1L,  1e2L,  1e3L,  1e4L,  1e5L,  1e6L,  1e7L,  1e8L,  1e9L,
      1e10L, 1e11L, 1e12L, 1e13L, 1e14L, 1e15L, 1e16L, 1e17L, 1e18L, 1e19L,
      1e20L, 1e21L, 1e22L, 1e23L, 1e24L, 1e25L, 1e26L, 1e27L, 1e28L, 1e29L,
      1e30L, 1e31L, 1e32L, 1e33L, 1e34L, 1e35L, 1e36L, 1e37L, 1e38L, 1e39L,
      1e40L, 1e41L, 1e42L, 1e43L, 1e44L, 1e45L, 1e46L, 1e47L, 1e48L};
  static constexpr int32_t EXACT_POWERS_OF_TEN = 48;
  static constexpr int32_t DIGITS_IN_MANTISSA = 33;
  static constexpr long double MAX_EXACT_INT =
      10384593717069655257060992658440191.0L;
};
#endif

// Take an exact mantissa and exponent and attempt to convert it using only
// exact floating point arithmetic. This only handles numbers with low
// exponents, but handles them quickly. This is an implementation of Clinger's
// Fast Path, as described above.
template <class T>
static inline bool
clinger_fast_path(typename fputil::FPBits<T>::UIntType mantissa, int32_t exp10,
                  typename fputil::FPBits<T>::UIntType *outputMantissa,
                  uint32_t *outputExp2) {
  if (mantissa >> fputil::FloatProperties<T>::MANTISSA_WIDTH > 0) {
    return false;
  }

  fputil::FPBits<T> result;
  T float_mantissa = static_cast<T>(mantissa);

  if (exp10 == 0) {
    result = fputil::FPBits<T>(float_mantissa);
  }
  if (exp10 > 0) {
    if (exp10 > ClingerConsts<T>::EXACT_POWERS_OF_TEN +
                    ClingerConsts<T>::DIGITS_IN_MANTISSA) {
      return false;
    }
    if (exp10 > ClingerConsts<T>::EXACT_POWERS_OF_TEN) {
      float_mantissa = float_mantissa *
                       ClingerConsts<T>::POWERS_OF_TEN_ARRAY
                           [exp10 - ClingerConsts<T>::EXACT_POWERS_OF_TEN];
      exp10 = ClingerConsts<T>::EXACT_POWERS_OF_TEN;
    }
    if (float_mantissa > ClingerConsts<T>::MAX_EXACT_INT) {
      return false;
    }
    result = fputil::FPBits<T>(float_mantissa *
                               ClingerConsts<T>::POWERS_OF_TEN_ARRAY[exp10]);
  } else if (exp10 < 0) {
    if (-exp10 > ClingerConsts<T>::EXACT_POWERS_OF_TEN) {
      return false;
    }
    result = fputil::FPBits<T>(float_mantissa /
                               ClingerConsts<T>::POWERS_OF_TEN_ARRAY[-exp10]);
  }
  *outputMantissa = result.get_mantissa();
  *outputExp2 = result.get_unbiased_exponent();
  return true;
}

// Takes a mantissa and base 10 exponent and converts it into its closest
// floating point type T equivalient. First we try the Eisel-Lemire algorithm,
// then if that fails then we fall back to a more accurate algorithm for
// accuracy. The resulting mantissa and exponent are placed in outputMantissa
// and outputExp2.
template <class T>
static inline void
decimal_exp_to_float(typename fputil::FPBits<T>::UIntType mantissa,
                     int32_t exp10, const char *__restrict numStart,
                     bool truncated,
                     typename fputil::FPBits<T>::UIntType *outputMantissa,
                     uint32_t *outputExp2) {
  // If the exponent is too large and can't be represented in this size of
  // float, return inf. These bounds are very loose, but are mostly serving as a
  // first pass. Some close numbers getting through is okay.
  if (exp10 >
      static_cast<int64_t>(fputil::FloatProperties<T>::EXPONENT_BIAS) / 3) {
    *outputMantissa = 0;
    *outputExp2 = fputil::FPBits<T>::MAX_EXPONENT;
    errno = ERANGE;
    return;
  }
  // If the exponent is too small even for a subnormal, return 0.
  if (exp10 < 0 &&
      -static_cast<int64_t>(exp10) >
          static_cast<int64_t>(fputil::FloatProperties<T>::EXPONENT_BIAS +
                               fputil::FloatProperties<T>::MANTISSA_WIDTH) /
              2) {
    *outputMantissa = 0;
    *outputExp2 = 0;
    errno = ERANGE;
    return;
  }

  if (!truncated) {
    if (clinger_fast_path<T>(mantissa, exp10, outputMantissa, outputExp2)) {
      return;
    }
  }

  // Try Eisel-Lemire
  if (eisel_lemire<T>(mantissa, exp10, outputMantissa, outputExp2)) {
    if (!truncated) {
      return;
    }
    // If the mantissa is truncated, then the result may be off by the LSB, so
    // check if rounding the mantissa up changes the result. If not, then it's
    // safe, else use the fallback.
    typename fputil::FPBits<T>::UIntType first_mantissa = *outputMantissa;
    uint32_t first_exp2 = *outputExp2;
    if (eisel_lemire<T>(mantissa + 1, exp10, outputMantissa, outputExp2)) {
      if (*outputMantissa == first_mantissa && *outputExp2 == first_exp2) {
        return;
      }
    }
  }

  simple_decimal_conversion<T>(numStart, outputMantissa, outputExp2);

  return;
}

// Takes a mantissa and base 2 exponent and converts it into its closest
// floating point type T equivalient. Since the exponent is already in the right
// form, this is mostly just shifting and rounding. This is used for hexadecimal
// numbers since a base 16 exponent multiplied by 4 is the base 2 exponent.
template <class T>
static inline void
binary_exp_to_float(typename fputil::FPBits<T>::UIntType mantissa, int32_t exp2,
                    bool truncated,
                    typename fputil::FPBits<T>::UIntType *outputMantissa,
                    uint32_t *outputExp2) {
  using BitsType = typename fputil::FPBits<T>::UIntType;

  // This is the number of leading zeroes a properly normalized float of type T
  // should have.
  constexpr int32_t NUMBITS = sizeof(BitsType) * 8;
  constexpr int32_t INF_EXP =
      (1 << fputil::FloatProperties<T>::EXPONENT_WIDTH) - 1;

  // Normalization step 1: Bring the leading bit to the highest bit of BitsType.
  uint32_t amount_to_shift_left = leading_zeroes<BitsType>(mantissa);
  mantissa <<= amount_to_shift_left;

  // Keep exp2 representing the exponent of the lowest bit of BitsType.
  exp2 -= amount_to_shift_left;

  // biasedExponent represents the biased exponent of the most significant bit.
  int32_t biased_exponent =
      exp2 + NUMBITS + fputil::FPBits<T>::EXPONENT_BIAS - 1;

  // Handle numbers that're too large and get squashed to inf
  if (biased_exponent >= INF_EXP) {
    // This indicates an overflow, so we make the result INF and set errno.
    *outputExp2 = (1 << fputil::FloatProperties<T>::EXPONENT_WIDTH) - 1;
    *outputMantissa = 0;
    errno = ERANGE;
    return;
  }

  uint32_t amount_to_shift_right =
      NUMBITS - fputil::FloatProperties<T>::MANTISSA_WIDTH - 1;

  // Handle subnormals.
  if (biased_exponent <= 0) {
    amount_to_shift_right += 1 - biased_exponent;
    biased_exponent = 0;

    if (amount_to_shift_right > NUMBITS) {
      // Return 0 if the exponent is too small.
      *outputMantissa = 0;
      *outputExp2 = 0;
      errno = ERANGE;
      return;
    }
  }

  BitsType round_bit_mask = BitsType(1) << (amount_to_shift_right - 1);
  BitsType sticky_mask = round_bit_mask - 1;
  bool round_bit = mantissa & round_bit_mask;
  bool sticky_bit = static_cast<bool>(mantissa & sticky_mask) || truncated;

  if (amount_to_shift_right < NUMBITS) {
    // Shift the mantissa and clear the implicit bit.
    mantissa >>= amount_to_shift_right;
    mantissa &= fputil::FloatProperties<T>::MANTISSA_MASK;
  } else {
    mantissa = 0;
  }
  bool least_significant_bit = mantissa & BitsType(1);
  // Perform rounding-to-nearest, tie-to-even.
  if (round_bit && (least_significant_bit || sticky_bit)) {
    ++mantissa;
  }

  if (mantissa > fputil::FloatProperties<T>::MANTISSA_MASK) {
    // Rounding causes the exponent to increase.
    ++biased_exponent;

    if (biased_exponent == INF_EXP) {
      errno = ERANGE;
    }
  }

  if (biased_exponent == 0) {
    errno = ERANGE;
  }

  *outputMantissa = mantissa & fputil::FloatProperties<T>::MANTISSA_MASK;
  *outputExp2 = biased_exponent;
}

// checks if the next 4 characters of the string pointer are the start of a
// hexadecimal floating point number. Does not advance the string pointer.
static inline bool is_float_hex_start(const char *__restrict src,
                                      const char decimalPoint) {
  if (!(*src == '0' && (*(src + 1) | 32) == 'x')) {
    return false;
  }
  if (*(src + 2) == decimalPoint) {
    return isalnum(*(src + 3)) && b36_char_to_int(*(src + 3)) < 16;
  } else {
    return isalnum(*(src + 2)) && b36_char_to_int(*(src + 2)) < 16;
  }
}

// Takes the start of a string representing a decimal float, as well as the
// local decimalPoint. It returns if it suceeded in parsing any digits, and if
// the return value is true then the outputs are pointer to the end of the
// number, and the mantissa and exponent for the closest float T representation.
// If the return value is false, then it is assumed that there is no number
// here.
template <class T>
static inline bool
decimal_string_to_float(const char *__restrict src, const char DECIMAL_POINT,
                        char **__restrict strEnd,
                        typename fputil::FPBits<T>::UIntType *outputMantissa,
                        uint32_t *outputExponent) {
  using BitsType = typename fputil::FPBits<T>::UIntType;
  constexpr uint32_t BASE = 10;
  constexpr char EXPONENT_MARKER = 'e';

  const char *__restrict num_start = src;
  bool truncated = false;
  bool seen_digit = false;
  bool after_decimal = false;
  BitsType mantissa = 0;
  int32_t exponent = 0;

  // The goal for the first step of parsing is to convert the number in src to
  // the format mantissa * (base ^ exponent)

  // The loop fills the mantissa with as many digits as it can hold
  const BitsType bitstype_max_div_by_base =
      cpp::numeric_limits<BitsType>::max() / BASE;
  while (true) {
    if (isdigit(*src)) {
      uint32_t digit = *src - '0';
      seen_digit = true;

      if (mantissa < bitstype_max_div_by_base) {
        mantissa = (mantissa * BASE) + digit;
        if (after_decimal) {
          --exponent;
        }
      } else {
        if (digit > 0)
          truncated = true;
        if (!after_decimal)
          ++exponent;
      }

      ++src;
      continue;
    }
    if (*src == DECIMAL_POINT) {
      if (after_decimal) {
        break; // this means that *src points to a second decimal point, ending
               // the number.
      }
      after_decimal = true;
      ++src;
      continue;
    }
    // The character is neither a digit nor a decimal point.
    break;
  }

  if (!seen_digit)
    return false;

  if ((*src | 32) == EXPONENT_MARKER) {
    if (*(src + 1) == '+' || *(src + 1) == '-' || isdigit(*(src + 1))) {
      ++src;
      char *temp_str_end;
      int32_t add_to_exponent = strtointeger<int32_t>(src, &temp_str_end, 10);
      if (add_to_exponent > 100000)
        add_to_exponent = 100000;
      else if (add_to_exponent < -100000)
        add_to_exponent = -100000;

      src = temp_str_end;
      exponent += add_to_exponent;
    }
  }

  *strEnd = const_cast<char *>(src);
  if (mantissa == 0) { // if we have a 0, then also 0 the exponent.
    *outputMantissa = 0;
    *outputExponent = 0;
  } else {
    decimal_exp_to_float<T>(mantissa, exponent, num_start, truncated,
                            outputMantissa, outputExponent);
  }
  return true;
}

// Takes the start of a string representing a hexadecimal float, as well as the
// local decimal point. It returns if it suceeded in parsing any digits, and if
// the return value is true then the outputs are pointer to the end of the
// number, and the mantissa and exponent for the closest float T representation.
// If the return value is false, then it is assumed that there is no number
// here.
template <class T>
static inline bool hexadecimal_string_to_float(
    const char *__restrict src, const char DECIMAL_POINT,
    char **__restrict strEnd,
    typename fputil::FPBits<T>::UIntType *outputMantissa,
    uint32_t *outputExponent) {
  using BitsType = typename fputil::FPBits<T>::UIntType;
  constexpr uint32_t BASE = 16;
  constexpr char EXPONENT_MARKER = 'p';

  bool truncated = false;
  bool seen_digit = false;
  bool after_decimal = false;
  BitsType mantissa = 0;
  int32_t exponent = 0;

  // The goal for the first step of parsing is to convert the number in src to
  // the format mantissa * (base ^ exponent)

  // The loop fills the mantissa with as many digits as it can hold
  const BitsType bitstype_max_div_by_base =
      cpp::numeric_limits<BitsType>::max() / BASE;
  while (true) {
    if (isalnum(*src)) {
      uint32_t digit = b36_char_to_int(*src);
      if (digit < BASE)
        seen_digit = true;
      else
        break;

      if (mantissa < bitstype_max_div_by_base) {
        mantissa = (mantissa * BASE) + digit;
        if (after_decimal)
          --exponent;
      } else {
        if (digit > 0)
          truncated = true;
        if (!after_decimal)
          ++exponent;
      }
      ++src;
      continue;
    }
    if (*src == DECIMAL_POINT) {
      if (after_decimal) {
        break; // this means that *src points to a second decimal point, ending
               // the number.
      }
      after_decimal = true;
      ++src;
      continue;
    }
    // The character is neither a hexadecimal digit nor a decimal point.
    break;
  }

  if (!seen_digit)
    return false;

  // Convert the exponent from having a base of 16 to having a base of 2.
  exponent *= 4;

  if ((*src | 32) == EXPONENT_MARKER) {
    if (*(src + 1) == '+' || *(src + 1) == '-' || isdigit(*(src + 1))) {
      ++src;
      char *temp_str_end;
      int32_t add_to_exponent = strtointeger<int32_t>(src, &temp_str_end, 10);
      if (add_to_exponent > 100000)
        add_to_exponent = 100000;
      else if (add_to_exponent < -100000)
        add_to_exponent = -100000;
      src = temp_str_end;
      exponent += add_to_exponent;
    }
  }
  *strEnd = const_cast<char *>(src);
  if (mantissa == 0) { // if we have a 0, then also 0 the exponent.
    *outputMantissa = 0;
    *outputExponent = 0;
  } else {
    binary_exp_to_float<T>(mantissa, exponent, truncated, outputMantissa,
                           outputExponent);
  }
  return true;
}

// Takes a pointer to a string and a pointer to a string pointer. This function
// is used as the backend for all of the string to float functions.
template <class T>
static inline T strtofloatingpoint(const char *__restrict src,
                                   char **__restrict strEnd) {
  using BitsType = typename fputil::FPBits<T>::UIntType;
  fputil::FPBits<T> result = fputil::FPBits<T>();
  const char *original_src = src;
  bool seen_digit = false;
  src = first_non_whitespace(src);

  if (*src == '+' || *src == '-') {
    if (*src == '-') {
      result.set_sign(true);
    }
    ++src;
  }

  static constexpr char DECIMAL_POINT = '.';
  static const char *inf_string = "infinity";
  static const char *nan_string = "nan";

  // bool truncated = false;

  if (isdigit(*src) || *src == DECIMAL_POINT) { // regular number
    int base = 10;
    if (is_float_hex_start(src, DECIMAL_POINT)) {
      base = 16;
      src += 2;
      seen_digit = true;
    }
    char *new_str_end = nullptr;

    BitsType output_mantissa = 0;
    uint32_t output_exponent = 0;
    if (base == 16) {
      seen_digit = hexadecimal_string_to_float<T>(
          src, DECIMAL_POINT, &new_str_end, &output_mantissa, &output_exponent);
    } else { // base is 10
      seen_digit = decimal_string_to_float<T>(
          src, DECIMAL_POINT, &new_str_end, &output_mantissa, &output_exponent);
    }

    if (seen_digit) {
      src += new_str_end - src;
      result.set_mantissa(output_mantissa);
      result.set_unbiased_exponent(output_exponent);
    }
  } else if ((*src | 32) == 'n') { // NaN
    if ((src[1] | 32) == nan_string[1] && (src[2] | 32) == nan_string[2]) {
      seen_digit = true;
      src += 3;
      BitsType nan_mantissa = 0;
      // this handles the case of `NaN(n-character-sequence)`, where the
      // n-character-sequence is made of 0 or more letters and numbers in any
      // order.
      if (*src == '(') {
        const char *left_paren = src;
        ++src;
        while (isalnum(*src))
          ++src;
        if (*src == ')') {
          ++src;
          char *temp_src = 0;
          if (isdigit(*(left_paren + 1))) {
            // This is to prevent errors when BitsType is larger than 64 bits,
            // since strtointeger only supports up to 64 bits. This is actually
            // more than is required by the specification, which says for the
            // input type "NAN(n-char-sequence)" that "the meaning of
            // the n-char sequence is implementation-defined."
            nan_mantissa = static_cast<BitsType>(
                strtointeger<uint64_t>(left_paren + 1, &temp_src, 0));
            if (*temp_src != ')')
              nan_mantissa = 0;
          }
        } else
          src = left_paren;
      }
      nan_mantissa |= fputil::FloatProperties<T>::QUIET_NAN_MASK;
      if (result.get_sign()) {
        result = fputil::FPBits<T>(result.build_quiet_nan(nan_mantissa));
        result.set_sign(true);
      } else {
        result.set_sign(false);
        result = fputil::FPBits<T>(result.build_quiet_nan(nan_mantissa));
      }
    }
  } else if ((*src | 32) == 'i') { // INF
    if ((src[1] | 32) == inf_string[1] && (src[2] | 32) == inf_string[2]) {
      seen_digit = true;
      if (result.get_sign())
        result = result.neg_inf();
      else
        result = result.inf();
      if ((src[3] | 32) == inf_string[3] && (src[4] | 32) == inf_string[4] &&
          (src[5] | 32) == inf_string[5] && (src[6] | 32) == inf_string[6] &&
          (src[7] | 32) == inf_string[7]) {
        // if the string is "INFINITY" then strEnd needs to be set to src + 8.
        src += 8;
      } else {
        src += 3;
      }
    }
  }
  if (!seen_digit) { // If there is nothing to actually parse, then return 0.
    if (strEnd != nullptr)
      *strEnd = const_cast<char *>(original_src);
    return T(0);
  }

  if (strEnd != nullptr)
    *strEnd = const_cast<char *>(src);

  // This function only does something if T is long double and the platform uses
  // special 80 bit long doubles. Otherwise it should be inlined out.
  set_implicit_bit<T>(result);

  return T(result);
}

} // namespace internal
} // namespace __llvm_libc

#endif // LIBC_SRC_SUPPORT_STR_TO_FLOAT_H
