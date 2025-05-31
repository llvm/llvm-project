/*===-- include/flang/Decimal/decimal.h ---------------------------*- C++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

/* C and C++ API for binary-to/from-decimal conversion package. */

#ifndef FORTRAN_DECIMAL_DECIMAL_H_
#define FORTRAN_DECIMAL_DECIMAL_H_

#include "flang/Common/api-attrs.h"
#include <stddef.h>

#ifdef __cplusplus
// Binary-to-decimal conversions (formatting) produce a sequence of decimal
// digit characters in a NUL-terminated user-supplied buffer that constitute
// a decimal fraction (or zero), accompanied by a decimal exponent that
// you'll get to adjust and format yourself.  There can be a leading sign
// character.
// Negative zero is "-0".  The result can also be "NaN", "Inf", "+Inf",
// or "-Inf".
// If the conversion can't fit in the user-supplied buffer, a null pointer
// is returned.

#include "binary-floating-point.h"
namespace Fortran::decimal {
#endif /* C++ */

enum ConversionResultFlags {
  Exact = 0,
  Overflow = 1,
  Inexact = 2,
  Invalid = 4,
  Underflow = 8,
};

struct ConversionToDecimalResult {
  const char *str; /* may not be original buffer pointer; null if overflow */
  size_t length; /* does not include NUL terminator */
  int decimalExponent; /* assuming decimal point to the left of first digit */
  enum ConversionResultFlags flags;
};

/* The "minimize" flag causes the fewest number of output digits
 * to be emitted such that reading them back into the same binary
 * floating-point format with RoundNearest will return the same
 * value.
 */
enum DecimalConversionFlags {
  Minimize = 1, /* Minimize # of digits */
  AlwaysSign = 2, /* emit leading '+' if not negative */
};

/*
 * When allocating decimal conversion output buffers, use the maximum
 * number of significant decimal digits in the representation of the
 * least nonzero value, and add this extra space for a sign, a NUL, and
 * some extra due to the library working internally in base 10**16
 * and computing its output size in multiples of 16.
 */
#define EXTRA_DECIMAL_CONVERSION_SPACE (1 + 1 + 2 * 16 - 1)

#ifdef __cplusplus
template <int PREC>
RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal(char *, size_t,
    DecimalConversionFlags, int digits, enum FortranRounding rounding,
    BinaryFloatingPointNumber<PREC> x);

extern template RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal<8>(
    char *, size_t, enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<8>);
extern template RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal<11>(
    char *, size_t, enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<11>);
extern template RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal<24>(
    char *, size_t, enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<24>);
extern template RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal<53>(
    char *, size_t, enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<53>);
extern template RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal<64>(
    char *, size_t, enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<64>);
extern template RT_API_ATTRS ConversionToDecimalResult ConvertToDecimal<113>(
    char *, size_t, enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<113>);

template <int PREC> struct ConversionToBinaryResult {
  BinaryFloatingPointNumber<PREC> binary;
  enum ConversionResultFlags flags { Exact };
};

template <int PREC>
RT_API_ATTRS ConversionToBinaryResult<PREC> ConvertToBinary(const char *&,
    enum FortranRounding = RoundNearest, const char *end = nullptr);

extern template RT_API_ATTRS ConversionToBinaryResult<8> ConvertToBinary<8>(
    const char *&, enum FortranRounding, const char *end);
extern template RT_API_ATTRS ConversionToBinaryResult<11> ConvertToBinary<11>(
    const char *&, enum FortranRounding, const char *end);
extern template RT_API_ATTRS ConversionToBinaryResult<24> ConvertToBinary<24>(
    const char *&, enum FortranRounding, const char *end);
extern template RT_API_ATTRS ConversionToBinaryResult<53> ConvertToBinary<53>(
    const char *&, enum FortranRounding, const char *end);
extern template RT_API_ATTRS ConversionToBinaryResult<64> ConvertToBinary<64>(
    const char *&, enum FortranRounding, const char *end);
extern template RT_API_ATTRS ConversionToBinaryResult<113> ConvertToBinary<113>(
    const char *&, enum FortranRounding, const char *end);
} // namespace Fortran::decimal
extern "C" {
#define NS(x) Fortran::decimal::x
#else /* C++ */
#define NS(x) x
#endif /* C++ */

RT_API_ATTRS struct NS(ConversionToDecimalResult)
    ConvertFloatToDecimal(char *, size_t, enum NS(DecimalConversionFlags),
        int digits, enum NS(FortranRounding), float);
RT_API_ATTRS struct NS(ConversionToDecimalResult)
    ConvertDoubleToDecimal(char *, size_t, enum NS(DecimalConversionFlags),
        int digits, enum NS(FortranRounding), double);
RT_API_ATTRS struct NS(ConversionToDecimalResult)
    ConvertLongDoubleToDecimal(char *, size_t, enum NS(DecimalConversionFlags),
        int digits, enum NS(FortranRounding), long double);

RT_API_ATTRS enum NS(ConversionResultFlags)
    ConvertDecimalToFloat(const char **, float *, enum NS(FortranRounding));
RT_API_ATTRS enum NS(ConversionResultFlags)
    ConvertDecimalToDouble(const char **, double *, enum NS(FortranRounding));
RT_API_ATTRS enum NS(ConversionResultFlags) ConvertDecimalToLongDouble(
    const char **, long double *, enum NS(FortranRounding));
#undef NS
#ifdef __cplusplus
} // extern "C"
#endif
#endif
