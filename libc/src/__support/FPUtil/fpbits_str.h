//===------ Pretty print function for FPBits --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_STR_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_STR_H

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/FloatProperties.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc {

// Converts the bits to a string in the following format:
//    "0x<NNN...N> = S: N, E: 0xNNNN, M:0xNNN...N"
// 1. N is a hexadecimal digit.
// 2. The hexadecimal number on the LHS is the raw numerical representation
//    of the bits.
// 3. The exponent is always 16 bits wide irrespective of the type of the
//    floating encoding.
template <typename T> LIBC_INLINE cpp::string str(fputil::FPBits<T> x) {
  using UIntType = typename fputil::FPBits<T>::UIntType;

  if (x.is_nan())
    return "(NaN)";
  if (x.is_inf())
    return x.get_sign() ? "(-Infinity)" : "(+Infinity)";

  auto zerofill = [](char *arr, size_t n) {
    for (size_t i = 0; i < n; ++i)
      arr[i] = '0';
  };

  cpp::string s("0x");
  char bitsbuf[IntegerToString::hex_bufsize<UIntType>()];
  zerofill(bitsbuf, sizeof(bitsbuf));
  IntegerToString::hex(x.bits, bitsbuf, false);
  s += cpp::string(bitsbuf, sizeof(bitsbuf));

  s += " = (";
  s += cpp::string("S: ") + (x.get_sign() ? "1" : "0");

  char expbuf[IntegerToString::hex_bufsize<uint16_t>()];
  zerofill(expbuf, sizeof(expbuf));
  IntegerToString::hex(x.get_unbiased_exponent(), expbuf, false);
  s += cpp::string(", E: 0x") + cpp::string(expbuf, sizeof(expbuf));

  if constexpr (cpp::is_same_v<T, long double> &&
                fputil::FloatProperties<long double>::MANTISSA_WIDTH == 63) {
    s += cpp::string(", I: ") + (x.get_implicit_bit() ? "1" : "0");
  }

  char mantbuf[IntegerToString::hex_bufsize<UIntType>()] = {'0'};
  zerofill(mantbuf, sizeof(mantbuf));
  IntegerToString::hex(x.get_mantissa(), mantbuf, false);
  s += cpp::string(", M: 0x") + cpp::string(mantbuf, sizeof(mantbuf));

  s += ")";
  return s;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_STR_H
