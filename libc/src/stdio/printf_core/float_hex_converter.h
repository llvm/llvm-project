//===-- Hexadecimal Converter for printf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_HEX_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_HEX_CONVERTER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/float_inf_nan_converter.h"
#include "src/stdio/printf_core/writer.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

using MantissaInt = fputil::FPBits<long double>::UIntType;

LIBC_INLINE int convert_float_hex_exp(Writer *writer,
                                      const FormatSection &to_conv) {
  // All of the letters will be defined relative to variable a, which will be
  // the appropriate case based on the name of the conversion. This converts any
  // conversion name into the letter 'a' with the appropriate case.
  const char a = (to_conv.conv_name & 32) | 'A';

  bool is_negative;
  int exponent;
  MantissaInt mantissa;
  bool is_inf_or_nan;
  uint32_t mantissa_width;
  int exponent_bias;
  if (to_conv.length_modifier == LengthModifier::L) {
    mantissa_width = fputil::MantissaWidth<long double>::VALUE;
    exponent_bias = fputil::FPBits<long double>::EXPONENT_BIAS;
    fputil::FPBits<long double>::UIntType float_raw = to_conv.conv_val_raw;
    fputil::FPBits<long double> float_bits(float_raw);
    is_negative = float_bits.get_sign();
    exponent = float_bits.get_exponent();
    mantissa = float_bits.get_explicit_mantissa();
    is_inf_or_nan = float_bits.is_inf_or_nan();
  } else {
    mantissa_width = fputil::MantissaWidth<double>::VALUE;
    exponent_bias = fputil::FPBits<double>::EXPONENT_BIAS;
    fputil::FPBits<double>::UIntType float_raw =
        static_cast<fputil::FPBits<double>::UIntType>(to_conv.conv_val_raw);
    fputil::FPBits<double> float_bits(float_raw);
    is_negative = float_bits.get_sign();
    exponent = float_bits.get_exponent();
    mantissa = float_bits.get_explicit_mantissa();
    is_inf_or_nan = float_bits.is_inf_or_nan();
  }

  if (is_inf_or_nan)
    return convert_inf_nan(writer, to_conv);

  char sign_char = 0;

  if (is_negative)
    sign_char = '-';
  else if ((to_conv.flags & FormatFlags::FORCE_SIGN) == FormatFlags::FORCE_SIGN)
    sign_char = '+'; // FORCE_SIGN has precedence over SPACE_PREFIX
  else if ((to_conv.flags & FormatFlags::SPACE_PREFIX) ==
           FormatFlags::SPACE_PREFIX)
    sign_char = ' ';

  // Handle the exponent for numbers with a 0 exponent
  if (exponent == -exponent_bias) {
    if (mantissa > 0) // Subnormals
      ++exponent;
    else // Zeroes
      exponent = 0;
  }

  constexpr size_t BITS_IN_HEX_DIGIT = 4;

  // This is to handle situations where the mantissa isn't an even number of hex
  // digits. This is primarily relevant for x86 80 bit long doubles, which have
  // 63 bit mantissas.
  if (mantissa_width % BITS_IN_HEX_DIGIT != 0) {
    exponent -= mantissa_width % BITS_IN_HEX_DIGIT;
  }

  // This is the max number of digits it can take to represent the mantissa.
  // Since the number is in bits, we divide by 4, and then add one to account
  // for the extra implicit bit. We use the larger of the two possible values
  // since the size must be constant.
  constexpr size_t MANT_BUFF_LEN =
      (fputil::MantissaWidth<long double>::VALUE / BITS_IN_HEX_DIGIT) + 1;
  char mant_buffer[MANT_BUFF_LEN];

  size_t mant_len = (mantissa_width / BITS_IN_HEX_DIGIT) + 1;

  // Precision only tracks the number of digits after the hexadecimal point, so
  // we have to add one to account for the digit before the hexadecimal point.
  if (to_conv.precision + 1 < static_cast<int>(mant_len) &&
      to_conv.precision + 1 > 0) {
    const size_t intended_digits = to_conv.precision + 1;
    const size_t shift_amount =
        (mant_len - intended_digits) * BITS_IN_HEX_DIGIT;

    const MantissaInt truncated_bits =
        mantissa & ((MantissaInt(1) << shift_amount) - 1);
    const MantissaInt halfway_const = MantissaInt(1) << (shift_amount - 1);

    mantissa >>= shift_amount;

    switch (fputil::quick_get_round()) {
    case FE_TONEAREST:
      // Round to nearest, if it's exactly halfway then round to even.
      if (truncated_bits > halfway_const)
        ++mantissa;
      else if (truncated_bits == halfway_const)
        mantissa = mantissa + (mantissa & 1);
      break;
    case FE_DOWNWARD:
      if (truncated_bits > 0 && is_negative)
        ++mantissa;
      break;
    case FE_UPWARD:
      if (truncated_bits > 0 && !is_negative)
        ++mantissa;
      break;
    case FE_TOWARDZERO:
      break;
    }

    // If the rounding caused an overflow, shift the mantissa and adjust the
    // exponent to match.
    if (mantissa >= (MantissaInt(1) << (intended_digits * BITS_IN_HEX_DIGIT))) {
      mantissa >>= BITS_IN_HEX_DIGIT;
      exponent += BITS_IN_HEX_DIGIT;
    }

    mant_len = intended_digits;
  }

  size_t mant_cur = mant_len;
  size_t first_non_zero = 1;
  for (; mant_cur > 0; --mant_cur, mantissa >>= 4) {
    char mant_mod_16 = static_cast<char>(mantissa) & 15;
    char new_digit =
        (mant_mod_16 > 9) ? (mant_mod_16 - 10 + a) : (mant_mod_16 + '0');
    mant_buffer[mant_cur - 1] = new_digit;
    if (new_digit != '0' && first_non_zero < mant_cur)
      first_non_zero = mant_cur;
  }

  size_t mant_digits = first_non_zero;
  if (to_conv.precision >= 0)
    mant_digits = mant_len;

  // This approximates the number of digits it will take to represent the
  // exponent. The calculation is ceil((bits * 5) / 16). Floor also works, but
  // only on exact multiples of 16. We add 1 for the sign.
  // Relevant sizes:
  // 15 -> 5
  // 11 -> 4
  // 8  -> 3
  constexpr size_t EXP_LEN =
      (((fputil::ExponentWidth<long double>::VALUE * 5) + 15) / 16) + 1;
  char exp_buffer[EXP_LEN];

  bool exp_is_negative = false;
  if (exponent < 0) {
    exp_is_negative = true;
    exponent = -exponent;
  }

  size_t exp_cur = EXP_LEN;
  for (; exponent > 0; --exp_cur, exponent /= 10) {
    exp_buffer[exp_cur - 1] = (exponent % 10) + '0';
  }
  if (exp_cur == EXP_LEN) { // if nothing else was written, write a 0.
    exp_buffer[EXP_LEN - 1] = '0';
    exp_cur = EXP_LEN - 1;
  }

  exp_buffer[exp_cur - 1] = exp_is_negative ? '-' : '+';
  --exp_cur;

  // these are signed to prevent underflow due to negative values. The eventual
  // values will always be non-negative.
  int trailing_zeroes = 0;
  int padding;

  // prefix is "0x", and always appears.
  constexpr size_t PREFIX_LEN = 2;
  char prefix[PREFIX_LEN];
  prefix[0] = '0';
  prefix[1] = a + ('x' - 'a');
  const cpp::string_view prefix_str(prefix, PREFIX_LEN);

  // If the precision is greater than the actual result, pad with 0s
  if (to_conv.precision > static_cast<int>(mant_digits - 1))
    trailing_zeroes = to_conv.precision - (mant_digits - 1);

  bool has_hexadecimal_point =
      (mant_digits > 1) || ((to_conv.flags & FormatFlags::ALTERNATE_FORM) ==
                            FormatFlags::ALTERNATE_FORM);
  constexpr cpp::string_view HEXADECIMAL_POINT(".");

  // This is for the letter 'p' before the exponent.
  const char exp_seperator = a + ('p' - 'a');
  constexpr int EXP_SEPERATOR_LEN = 1;

  padding = to_conv.min_width - (sign_char > 0 ? 1 : 0) - PREFIX_LEN -
            mant_digits - (has_hexadecimal_point ? 1 : 0) - EXP_SEPERATOR_LEN -
            (EXP_LEN - exp_cur);
  if (padding < 0)
    padding = 0;

  if ((to_conv.flags & FormatFlags::LEFT_JUSTIFIED) ==
      FormatFlags::LEFT_JUSTIFIED) {
    // The pattern is (sign), 0x, digit, (.), (other digits), (zeroes), p,
    // exponent, (spaces)
    if (sign_char > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(sign_char));
    RET_IF_RESULT_NEGATIVE(writer->write(prefix_str));
    RET_IF_RESULT_NEGATIVE(writer->write(mant_buffer[0]));
    if (has_hexadecimal_point)
      RET_IF_RESULT_NEGATIVE(writer->write(HEXADECIMAL_POINT));
    if (mant_digits > 1)
      RET_IF_RESULT_NEGATIVE(writer->write({mant_buffer + 1, mant_digits - 1}));
    if (trailing_zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write('0', trailing_zeroes));
    RET_IF_RESULT_NEGATIVE(writer->write(exp_seperator));
    RET_IF_RESULT_NEGATIVE(
        writer->write({exp_buffer + exp_cur, EXP_LEN - exp_cur}));
    if (padding > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(' ', padding));
  } else {
    // The pattern is (spaces), (sign), 0x, (zeroes), digit, (.), (other
    // digits), (zeroes), p, exponent
    if ((padding > 0) && ((to_conv.flags & FormatFlags::LEADING_ZEROES) !=
                          FormatFlags::LEADING_ZEROES))
      RET_IF_RESULT_NEGATIVE(writer->write(' ', padding));
    if (sign_char > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(sign_char));
    RET_IF_RESULT_NEGATIVE(writer->write(prefix_str));
    if ((padding > 0) && ((to_conv.flags & FormatFlags::LEADING_ZEROES) ==
                          FormatFlags::LEADING_ZEROES))
      RET_IF_RESULT_NEGATIVE(writer->write('0', padding));
    RET_IF_RESULT_NEGATIVE(writer->write(mant_buffer[0]));
    if (has_hexadecimal_point)
      RET_IF_RESULT_NEGATIVE(writer->write(HEXADECIMAL_POINT));
    if (mant_digits > 1)
      RET_IF_RESULT_NEGATIVE(writer->write({mant_buffer + 1, mant_digits - 1}));
    if (trailing_zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write('0', trailing_zeroes));
    RET_IF_RESULT_NEGATIVE(writer->write(exp_seperator));
    RET_IF_RESULT_NEGATIVE(
        writer->write({exp_buffer + exp_cur, EXP_LEN - exp_cur}));
  }
  return WRITE_OK;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_HEX_CONVERTER_H
