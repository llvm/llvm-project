//===-- Implementation header for strfromx() utilitites -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// According to the C23 standard, any input character sequences except a
// precision specifier and the usual floating point formats, namely
// %{a,A,e,E,f,F,g,G}, are not allowed and any code that does otherwise results
// in undefined behaviour(including use of a '%%' conversion specifier); which
// in this case is that the buffer string is simply populated with the format
// string. The case of the input being nullptr should be handled in the calling
// function (strfromf, strfromd, strfroml) itself.

#ifndef LLVM_LIBC_SRC_STDLIB_STRFROM_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_STRFROM_UTIL_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/converter_atlas.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/writer.h"
#include "src/__support/str_to_integer.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

template <typename T>
using storage_type = typename fputil::FPBits<T>::StorageType;

template <typename T, printf_core::OverflowMode overflow_mode>
LIBC_INLINE int strfromfloat_convert(printf_core::Writer<overflow_mode> *writer,
                                     const char *__restrict format, T fp) {
  printf_core::FormatSection section;
  size_t cur_pos = 0;

  if (format[cur_pos] == '%') {
    section.has_conv = true;
    ++cur_pos;

    // handle precision
    section.precision = -1;
    if (format[cur_pos] == '.') {
      ++cur_pos;
      section.precision = 0;

      // The standard does not allow the '*' (asterisk) operator for strfromx()
      // functions
      if (internal::isdigit(format[cur_pos])) {
        auto result = internal::strtointeger<int>(format + cur_pos, 10);
        section.precision += result.value;
        cur_pos += result.parsed_len;
      }
    }

    section.conv_name = format[cur_pos];
    switch (format[cur_pos]) {
    case 'a':
    case 'A':
    case 'e':
    case 'E':
    case 'f':
    case 'F':
    case 'g':
    case 'G':
      break;
    default:
      section.has_conv = false;
      break;
    }
  } else {
    section.has_conv = false;
  }

  if (!section.has_conv)
    return writer->write(format);

  fputil::FPBits<T> strfromfloat_bits(fp);
  if (strfromfloat_bits.is_inf_or_nan())
    return convert_inf_nan(
        writer,
        printf_core::InfNanFPBitsProperties{
            .is_negative = strfromfloat_bits.is_neg(),
            .mantissa_is_zero = strfromfloat_bits.get_mantissa() == 0,
        },
        section);

  switch (section.conv_name) {
  case 'f':
  case 'F':
    return printf_core::convert_finite_float_decimal_typed(writer, section,
                                                           strfromfloat_bits);
  case 'e':
  case 'E':
    return printf_core::convert_finite_float_dec_exp_typed(writer, section,
                                                           strfromfloat_bits);
  case 'a':
  case 'A':
    // There is no typed conversion function to convert single precision float
    // to hex exponential format, and the convert_finite_float_hex_exp()
    // requires a double or long double value to work correctly.
    if constexpr (cpp::is_same_v<T, float>) {
      return printf_core::convert_finite_float_hex_exp(
          writer,
          printf_core::get_float_hex_exp_fp_bits_properties_typed(
              fputil::FPBits<double>(static_cast<double>(fp))),
          section);
    } else {
      return printf_core::convert_finite_float_hex_exp(
          writer,
          printf_core::get_float_hex_exp_fp_bits_properties_typed(
              strfromfloat_bits),
          section);
    }
  case 'g':
  case 'G':
    return printf_core::convert_finite_float_dec_auto_typed(writer, section,
                                                            strfromfloat_bits);
  }
  __builtin_unreachable();
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_STRFROM_UTIL_H
