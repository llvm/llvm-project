//===-- String to float conversion utils ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_SUPPORT_STR_TO_FLOAT_H
#define LIBC_SRC_SUPPORT_STR_TO_FLOAT_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/detailed_powers_of_ten.h"
#include "src/__support/high_precision_decimal.h"
#include "src/__support/str_conv_utils.h"
#include "utils/CPP/Limits.h"
#include <errno.h>

namespace __llvm_libc {
namespace internal {

// Shifts right and rounds according to the following rules:
// 1) If the part being cut off is more than 2^(amountToShift - 1) then round
// up
// 2) If it is less than that number then round down
// 3) If it is exactly that number, then round so that the final number will be
// even
template <class T>
static inline T shiftRightAndRound(T numToShift, unsigned int amountToShift) {
  T result = numToShift >> amountToShift;
  T truncated = numToShift & ((1 << amountToShift) - 1);

  if (truncated < (1 << (amountToShift - 1))) {
    return result;
  } else if (truncated > (1 << (amountToShift - 1))) {
    return result + 1;
  } else {
    return result + (result & 1); // This rounds towards even.
  }
}

template <class T> uint32_t leadingZeroes(T inputNumber) {
  // TODO(michaelrj): investigate the portability of using something like
  // __builtin_clz for specific types.
  constexpr uint32_t bitsInT = sizeof(T) * 8;
  if (inputNumber == 0) {
    return bitsInT;
  }
  uint32_t curGuess = bitsInT / 2;
  uint32_t rangeSize = bitsInT / 2;
  // while either shifting by curGuess does not get rid of all of the bits or
  // shifting by one less also gets rid of all of the bits then we have not
  // found the first bit.
  while (((inputNumber >> curGuess) > 0) ||
         ((inputNumber >> (curGuess - 1)) == 0)) {
    // Binary search for the first set bit
    rangeSize /= 2;
    if (rangeSize == 0) {
      break;
    }
    if ((inputNumber >> curGuess) > 0) {
      curGuess += rangeSize;
    } else {
      curGuess -= rangeSize;
    }
  }
  if (inputNumber >> curGuess > 0) {
    curGuess++;
  }
  return bitsInT - curGuess;
}

static inline uint64_t low64(__uint128_t num) {
  return static_cast<uint64_t>(num & 0xffffffffffffffff);
}

static inline uint64_t high64(__uint128_t num) {
  return static_cast<uint64_t>(num >> 64);
}

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
eiselLemire(typename fputil::FPBits<T>::UIntType mantissa, int32_t exp10,
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
  uint32_t clz = leadingZeroes<BitsType>(mantissa);
  mantissa <<= clz;

  uint32_t exp2 = exp10ToExp2(exp10) + BITS_IN_MANTISSA +
                  fputil::FloatProperties<T>::exponentBias - clz;

  // Multiplication
  const uint64_t *powerOfTen =
      DETAILED_POWERS_OF_TEN[exp10 - DETAILED_POWERS_OF_TEN_MIN_EXP_10];

  __uint128_t firstApprox = static_cast<__uint128_t>(mantissa) *
                            static_cast<__uint128_t>(powerOfTen[1]);

  // Wider Approximation
  __uint128_t finalApprox;
  // The halfway constant is used to check if the bits that will be shifted away
  // intially are all 1. For doubles this is 64 (bitstype size) - 52 (final
  // mantissa size) - 3 (we shift away the last two bits separately for
  // accuracy, and the most significant bit is ignored.) = 9. Similarly, it's 6
  // for floats in this case.
  const uint64_t halfwayConstant = sizeof(T) == 8 ? 0x1FF : 0x3F;
  if ((high64(firstApprox) & halfwayConstant) == halfwayConstant &&
      low64(firstApprox) + mantissa < mantissa) {
    __uint128_t lowBits = static_cast<__uint128_t>(mantissa) *
                          static_cast<__uint128_t>(powerOfTen[0]);
    __uint128_t secondApprox =
        firstApprox + static_cast<__uint128_t>(high64(lowBits));

    if ((high64(secondApprox) & halfwayConstant) == halfwayConstant &&
        low64(secondApprox) + 1 == 0 && low64(lowBits) + mantissa < mantissa) {
      return false;
    }
    finalApprox = secondApprox;
  } else {
    finalApprox = firstApprox;
  }

  // Shifting to 54 bits for doubles and 25 bits for floats
  BitsType msb = high64(finalApprox) >> (BITS_IN_MANTISSA - 1);
  BitsType finalMantissa =
      high64(finalApprox) >> (msb + BITS_IN_MANTISSA -
                              (fputil::FloatProperties<T>::mantissaWidth + 3));
  exp2 -= 1 ^ msb; // same as !msb

  // Half-way ambiguity
  if (low64(finalApprox) == 0 && (high64(finalApprox) & halfwayConstant) == 0 &&
      (finalMantissa & 3) == 1) {
    return false;
  }

  // From 54 to 53 bits for doubles and 25 to 24 bits for floats
  finalMantissa += finalMantissa & 1;
  finalMantissa >>= 1;
  if ((finalMantissa >> (fputil::FloatProperties<T>::mantissaWidth + 1)) > 0) {
    finalMantissa >>= 1;
    ++exp2;
  }

  // The if block is equivalent to (but has fewer branches than):
  //   if exp2 <= 0 || exp2 >= 0x7FF { etc }
  if (exp2 - 1 >= (1 << fputil::FloatProperties<T>::exponentWidth) - 2) {
    return false;
  }

  *outputMantissa = finalMantissa;
  *outputExp2 = exp2;
  return true;
}

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
simpleDecimalConversion(const char *__restrict numStart,
                        typename fputil::FPBits<T>::UIntType *outputMantissa,
                        uint32_t *outputExp2) {

  int32_t exp2 = 0;
  HighPrecsisionDecimal hpd = HighPrecsisionDecimal(numStart);

  if (hpd.getNumDigits() == 0) {
    *outputMantissa = 0;
    *outputExp2 = 0;
    return;
  }

  // If the exponent is too large and can't be represented in this size of
  // float, return inf.
  if (hpd.getDecimalPoint() > 0 &&
      exp10ToExp2(hpd.getDecimalPoint() - 1) >
          static_cast<int32_t>(fputil::FloatProperties<T>::exponentBias)) {
    *outputMantissa = 0;
    *outputExp2 = fputil::FPBits<T>::maxExponent;
    errno = ERANGE; // NOLINT
    return;
  }
  // If the exponent is too small even for a subnormal, return 0.
  if (hpd.getDecimalPoint() < 0 &&
      exp10ToExp2(-hpd.getDecimalPoint()) >
          static_cast<int32_t>(fputil::FloatProperties<T>::exponentBias +
                               fputil::FloatProperties<T>::mantissaWidth)) {
    *outputMantissa = 0;
    *outputExp2 = 0;
    errno = ERANGE; // NOLINT
    return;
  }

  // Right shift until the number is smaller than 1.
  while (hpd.getDecimalPoint() > 0) {
    int32_t shiftAmount = 0;
    if (hpd.getDecimalPoint() >= NUM_POWERS_OF_TWO) {
      shiftAmount = 60;
    } else {
      shiftAmount = POWERS_OF_TWO[hpd.getDecimalPoint()];
    }
    exp2 += shiftAmount;
    hpd.shift(-shiftAmount);
  }

  // Left shift until the number is between 1/2 and 1
  while (hpd.getDecimalPoint() < 0 ||
         (hpd.getDecimalPoint() == 0 && hpd.getDigits()[0] < 5)) {
    int32_t shiftAmount = 0;

    if (-hpd.getDecimalPoint() >= NUM_POWERS_OF_TWO) {
      shiftAmount = 60;
    } else if (hpd.getDecimalPoint() != 0) {
      shiftAmount = POWERS_OF_TWO[-hpd.getDecimalPoint()];
    } else { // This handles the case of the number being between .1 and .5
      shiftAmount = 1;
    }
    exp2 -= shiftAmount;
    hpd.shift(shiftAmount);
  }

  // Left shift once so that the number is between 1 and 2
  --exp2;
  hpd.shift(1);

  // Get the biased exponent
  exp2 += fputil::FloatProperties<T>::exponentBias;

  // Handle the exponent being too large (and return inf).
  if (exp2 >= fputil::FPBits<T>::maxExponent) {
    *outputMantissa = 0;
    *outputExp2 = fputil::FPBits<T>::maxExponent;
    errno = ERANGE; // NOLINT
    return;
  }

  // Shift left to fill the mantissa
  hpd.shift(fputil::FloatProperties<T>::mantissaWidth);
  typename fputil::FPBits<T>::UIntType finalMantissa =
      hpd.roundToIntegerType<typename fputil::FPBits<T>::UIntType>();

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
    finalMantissa =
        hpd.roundToIntegerType<typename fputil::FPBits<T>::UIntType>();

    // Check if by shifting right we've caused this to round to a normal number.
    if ((finalMantissa >> fputil::FloatProperties<T>::mantissaWidth) != 0) {
      ++exp2;
    }
  }

  // Check if rounding added a bit, and shift down if that's the case.
  if (finalMantissa == typename fputil::FPBits<T>::UIntType(2)
                           << fputil::FloatProperties<T>::mantissaWidth) {
    finalMantissa >>= 1;
    ++exp2;
  }

  *outputMantissa = finalMantissa;
  *outputExp2 = exp2;
}

// Takes a mantissa and base 10 exponent and converts it into its closest
// floating point type T equivalient. First we try the Eisel-Lemire algorithm,
// then if that fails then we fall back to a more accurate algorithm for
// accuracy. The resulting mantissa and exponent are placed in outputMantissa
// and outputExp2.
template <class T>
static inline void
decimalExpToFloat(typename fputil::FPBits<T>::UIntType mantissa, int32_t exp10,
                  const char *__restrict numStart, bool truncated,
                  typename fputil::FPBits<T>::UIntType *outputMantissa,
                  uint32_t *outputExp2) {

  // TODO: Implement Clinger's fast path, as well as other shortcuts here.

  // Try Eisel-Lemire
  if (eiselLemire<T>(mantissa, exp10, outputMantissa, outputExp2)) {
    if (!truncated) {
      return;
    }
    // If the mantissa is truncated, then the result may be off by the LSB, so
    // check if rounding the mantissa up changes the result. If not, then it's
    // safe, else use the fallback.
    typename fputil::FPBits<T>::UIntType firstMantissa = *outputMantissa;
    uint32_t firstExp2 = *outputExp2;
    if (eiselLemire<T>(mantissa + 1, exp10, outputMantissa, outputExp2)) {
      if (*outputMantissa == firstMantissa && *outputExp2 == firstExp2) {
        return;
      }
    }
  }

  simpleDecimalConversion<T>(numStart, outputMantissa, outputExp2);

  return;
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

// Takes a pointer to a string and a pointer to a string pointer. This function
// is used as the backend for all of the string to float functions.
template <class T>
static inline T strtofloatingpoint(const char *__restrict src,
                                   char **__restrict strEnd) {
  using BitsType = typename fputil::FPBits<T>::UIntType;
  fputil::FPBits<T> result = fputil::FPBits<T>();
  const char *originalSrc = src;
  bool seenDigit = false;
  src = first_non_whitespace(src);

  if (*src == '+' || *src == '-') {
    if (*src == '-') {
      result.setSign(true);
    }
    ++src;
  }

  static constexpr char DECIMAL_POINT = '.';
  static const char *INF_STRING = "infinity";
  static const char *NAN_STRING = "nan";

  bool truncated = false;

  if (isdigit(*src) || *src == DECIMAL_POINT) { // regular number
    int base = 10;
    char exponentMarker = 'e';
    if (is_float_hex_start(src, DECIMAL_POINT)) {
      base = 16;
      src += 2;
      exponentMarker = 'p';
      seenDigit = true;
    }
    const char *__restrict numStart = src;
    bool afterDecimal = false;

    BitsType mantissa = 0;
    int32_t exponent = 0;

    // The goal for the first step of parsing is to convert the number in src to
    // the format mantissa * (base ^ exponent)

    constexpr BitsType MANTISSA_MAX =
        BitsType(1) << (fputil::FloatProperties<T>::mantissaWidth +
                        1); // The extra bit is to give space for the implicit 1
    const BitsType BITSTYPE_MAX_DIV_BY_BASE =
        __llvm_libc::cpp::NumericLimits<BitsType>::max() / base;
    while ((isalnum(*src) || *src == DECIMAL_POINT) &&
           mantissa < BITSTYPE_MAX_DIV_BY_BASE) {
      if (*src == DECIMAL_POINT && afterDecimal) {
        break; // this means that *src points to a second decimal point, ending
               // the number.
      } else if (*src == DECIMAL_POINT) {
        afterDecimal = true;
        ++src;
        continue;
      }
      int digit = b36_char_to_int(*src);
      if (digit >= base) {
        break;
      }

      mantissa = (mantissa * base) + digit;
      seenDigit = true;
      if (afterDecimal) {
        --exponent;
      }

      ++src;
    }

    // The second loop is to run through the remaining digits after we've filled
    // the mantissa.
    while (isalnum(*src) || *src == DECIMAL_POINT) {
      if (*src == DECIMAL_POINT && afterDecimal) {
        break; // this means that *src points to a second decimal point, ending
               // the number.
      } else if (*src == DECIMAL_POINT) {
        afterDecimal = true;
        ++src;
        continue;
      }
      int digit = b36_char_to_int(*src);
      if (digit >= base) {
        break;
      }

      if (digit > 0) {
        truncated = true;
      }

      if (!afterDecimal) {
        exponent++;
      }

      ++src;
    }

    // if our base is 16 then convert the exponent to base 2
    if (base == 16) {
      exponent *= 4;
    }

    if ((*src | 32) == exponentMarker) {
      if (*(src + 1) == '+' || *(src + 1) == '-' || isdigit(*(src + 1))) {
        ++src;
        char *tempStrEnd;
        int32_t add_to_exponent = strtointeger<int32_t>(src, &tempStrEnd, 10);
        src += tempStrEnd - src;
        exponent += add_to_exponent;
      }
    }

    if (mantissa == 0) { // if we have a 0, then also 0 the exponent.
      exponent = 0;
    } else if (base == 16) {

      // These two loops should normalize the number if we assume the decimal
      // point is after the bit at mantissaWidth.
      // For example if type T is a 32 bit float, this should result in a
      // mantissa with its most significant 1 being at bit 23.
      while (mantissa < (MANTISSA_MAX >> 1)) {
        mantissa = mantissa << 1;
        --exponent;
      }
      BitsType mantissaCopy = mantissa;
      unsigned int amountToShift = 0;
      while (mantissaCopy > MANTISSA_MAX) {
        mantissaCopy = mantissaCopy >> 1;
        ++amountToShift;
      }
      exponent += amountToShift;
      mantissa = shiftRightAndRound(mantissa, amountToShift);

      // Account for the fact that the mantissa represented an integer
      // previously, but now represents the fractional part of a normalized
      // number.
      exponent += fputil::FloatProperties<T>::mantissaWidth;

      int32_t biasedExponent = exponent + fputil::FPBits<T>::exponentBias;
      if (biasedExponent <= 0) {
        // handle subnormals here

        // the most mantissa is currently normalized, meaning that the msb is
        // one bit left of where the decimal point should go.
        amountToShift = 1;
        mantissaCopy = mantissa >> 1;
        while (biasedExponent < 0 && mantissaCopy > 0) {
          mantissaCopy = mantissaCopy >> 1;
          ++amountToShift;
          ++biasedExponent;
        }
        // If we cut off any bits to fit this number into a subnormal, then it's
        // out of range for this size of float.
        if ((mantissa & ((1 << amountToShift) - 1)) > 0) {
          errno = ERANGE; // NOLINT
        }
        mantissa = shiftRightAndRound(mantissa, amountToShift);
        if (mantissa == 0) {
          biasedExponent = 0;
        }
      } else if (biasedExponent > result.maxExponent) {
        // This indicates an overflow, so we make the result INF and set errno.
        biasedExponent = result.maxExponent;
        mantissa = 0;
        errno = ERANGE; // NOLINT
      }

      result.setUnbiasedExponent(biasedExponent);
      result.setMantissa(mantissa);
    } else { // base is 10
      BitsType outputMantissa = 0;
      uint32_t outputExponent = 0;
      decimalExpToFloat<T>(mantissa, exponent, numStart, truncated,
                           &outputMantissa, &outputExponent);
      result.setMantissa(outputMantissa);
      result.setUnbiasedExponent(outputExponent);
    }

  } else if ((*src | 32) == 'n') { // NaN
    if ((src[1] | 32) == NAN_STRING[1] && (src[2] | 32) == NAN_STRING[2]) {
      seenDigit = true;
      src += 3;
      BitsType NaNMantissa = 0;
      if (*src == '(') {
        char *tempSrc = 0;
        if (isdigit(*(src + 1)) || *(src + 1) == ')') {
          NaNMantissa = strtointeger<BitsType>(src + 1, &tempSrc, 0);
          if (*tempSrc != ')') {
            NaNMantissa = 0;
          } else {
            src = tempSrc + 1;
          }
        }
      }
      NaNMantissa |= fputil::FloatProperties<T>::quietNaNMask;
      if (result.getSign()) {
        result = fputil::FPBits<T>(result.buildNaN(NaNMantissa));
        result.setSign(true);
      } else {
        result.setSign(false);
        result = fputil::FPBits<T>(result.buildNaN(NaNMantissa));
      }
    }
  } else if ((*src | 32) == 'i') { // INF
    if ((src[1] | 32) == INF_STRING[1] && (src[2] | 32) == INF_STRING[2]) {
      seenDigit = true;
      if (result.getSign())
        result = result.negInf();
      else
        result = result.inf();
      if ((src[3] | 32) == INF_STRING[3] && (src[4] | 32) == INF_STRING[4] &&
          (src[5] | 32) == INF_STRING[5] && (src[6] | 32) == INF_STRING[6] &&
          (src[7] | 32) == INF_STRING[7]) {
        // if the string is "INFINITY" then strEnd needs to be set to src + 8.
        src += 8;
      } else {
        src += 3;
      }
    }
  }
  if (!seenDigit) { // If there is nothing to actually parse, then return 0.
    if (strEnd != nullptr)
      *strEnd = const_cast<char *>(originalSrc);
    return T(0);
  }

  if (strEnd != nullptr)
    *strEnd = const_cast<char *>(src);

  return T(result);
}

} // namespace internal
} // namespace __llvm_libc

#endif // LIBC_SRC_SUPPORT_STR_TO_FLOAT_H
