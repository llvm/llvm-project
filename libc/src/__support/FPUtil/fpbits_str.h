//===------ Pretty print function for FPBits --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_BITS_STR_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_BITS_STR_H

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/FloatProperties.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE {

namespace details {

// Format T as uppercase hexadecimal number with leading zeros.
template <typename T>
using ZeroPaddedHexFmt = IntegerToString<
    T, typename radix::Hex::WithWidth<(sizeof(T) * 2)>::WithPrefix::Uppercase>;

} // namespace details

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

  const auto sign_char = [](bool sign) -> char { return sign ? '1' : '0'; };

  cpp::string s;

  const details::ZeroPaddedHexFmt<UIntType> bits(x.bits);
  s += bits.view();

  s += " = (S: ";
  s += sign_char(x.get_sign());

  s += ", E: ";
  const details::ZeroPaddedHexFmt<uint16_t> exponent(x.get_unbiased_exponent());
  s += exponent.view();

  if constexpr (cpp::is_same_v<T, long double> &&
                fputil::FloatProperties<long double>::MANTISSA_WIDTH == 63) {
    s += ", I: ";
    s += sign_char(x.get_implicit_bit());
  }

  s += ", M: ";
  const details::ZeroPaddedHexFmt<UIntType> mantissa(x.get_mantissa());
  s += mantissa.view();

  s += ')';
  return s;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_BITS_STR_H
