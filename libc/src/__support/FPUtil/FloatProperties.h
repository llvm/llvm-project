//===-- Properties of floating point numbers --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOATPROPERTIES_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOATPROPERTIES_H

#include "src/__support/UInt128.h"
#include "src/__support/macros/properties/float.h" // LIBC_COMPILER_HAS_FLOAT128

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {

// The supported floating point types.
enum class FPType {
  IEEE754_Binary16,
  IEEE754_Binary32,
  IEEE754_Binary64,
  IEEE754_Binary128,
  X86_Binary80,
};

template <FPType> struct FPProperties {};
template <> struct FPProperties<FPType::IEEE754_Binary32> {
  typedef uint32_t BitsType;

  static constexpr uint32_t BIT_WIDTH = sizeof(BitsType) * 8;

  static constexpr uint32_t MANTISSA_WIDTH = 23;
  // The mantissa precision includes the implicit bit.
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr uint32_t EXPONENT_WIDTH = 8;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  static constexpr BitsType SIGN_MASK = BitsType(1)
                                        << (EXPONENT_WIDTH + MANTISSA_WIDTH);
  static constexpr BitsType EXPONENT_MASK = ~(SIGN_MASK | MANTISSA_MASK);
  static constexpr uint32_t EXPONENT_BIAS = 127;

  static constexpr BitsType EXP_MANT_MASK = MANTISSA_MASK + EXPONENT_MASK;
  static_assert(EXP_MANT_MASK == ~SIGN_MASK,
                "Exponent and mantissa masks are not as expected.");

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType QUIET_NAN_MASK = 0x00400000U;
};

template <> struct FPProperties<FPType::IEEE754_Binary64> {
  typedef uint64_t BitsType;

  static constexpr uint32_t BIT_WIDTH = sizeof(BitsType) * 8;

  static constexpr uint32_t MANTISSA_WIDTH = 52;
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr uint32_t EXPONENT_WIDTH = 11;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  static constexpr BitsType SIGN_MASK = BitsType(1)
                                        << (EXPONENT_WIDTH + MANTISSA_WIDTH);
  static constexpr BitsType EXPONENT_MASK = ~(SIGN_MASK | MANTISSA_MASK);
  static constexpr uint32_t EXPONENT_BIAS = 1023;

  static constexpr BitsType EXP_MANT_MASK = MANTISSA_MASK + EXPONENT_MASK;
  static_assert(EXP_MANT_MASK == ~SIGN_MASK,
                "Exponent and mantissa masks are not as expected.");

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType QUIET_NAN_MASK = 0x0008000000000000ULL;
};

// Properties for numbers represented in 80 bits long double on non-Windows x86
// platforms.
template <> struct FPProperties<FPType::X86_Binary80> {
  typedef UInt128 BitsType;

  static constexpr uint32_t BIT_WIDTH = (sizeof(BitsType) * 8) - 48;
  static constexpr BitsType FULL_WIDTH_MASK = ((BitsType(1) << BIT_WIDTH) - 1);

  static constexpr uint32_t MANTISSA_WIDTH = 63;
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr uint32_t EXPONENT_WIDTH = 15;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;

  // The x86 80 bit float represents the leading digit of the mantissa
  // explicitly. This is the mask for that bit.
  static constexpr BitsType EXPLICIT_BIT_MASK = (BitsType(1) << MANTISSA_WIDTH);

  static constexpr BitsType SIGN_MASK =
      BitsType(1) << (EXPONENT_WIDTH + MANTISSA_WIDTH + 1);
  static constexpr BitsType EXPONENT_MASK =
      ((BitsType(1) << EXPONENT_WIDTH) - 1) << (MANTISSA_WIDTH + 1);
  static constexpr uint32_t EXPONENT_BIAS = 16383;

  static constexpr BitsType EXP_MANT_MASK =
      MANTISSA_MASK | EXPLICIT_BIT_MASK | EXPONENT_MASK;
  static_assert(EXP_MANT_MASK == (~SIGN_MASK & FULL_WIDTH_MASK),
                "Exponent and mantissa masks are not as expected.");

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType QUIET_NAN_MASK = BitsType(1)
                                             << (MANTISSA_WIDTH - 1);
};

// Properties for numbers represented in 128 bits long double on non x86
// platform.
template <> struct FPProperties<FPType::IEEE754_Binary128> {
  typedef UInt128 BitsType;

  static constexpr uint32_t BIT_WIDTH = sizeof(BitsType) << 3;

  static constexpr uint32_t MANTISSA_WIDTH = 112;
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr uint32_t EXPONENT_WIDTH = 15;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  static constexpr BitsType SIGN_MASK = BitsType(1)
                                        << (EXPONENT_WIDTH + MANTISSA_WIDTH);
  static constexpr BitsType EXPONENT_MASK = ~(SIGN_MASK | MANTISSA_MASK);
  static constexpr uint32_t EXPONENT_BIAS = 16383;

  static constexpr BitsType EXP_MANT_MASK = MANTISSA_MASK | EXPONENT_MASK;
  static_assert(EXP_MANT_MASK == ~SIGN_MASK,
                "Exponent and mantissa masks are not as expected.");

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType QUIET_NAN_MASK = BitsType(1)
                                             << (MANTISSA_WIDTH - 1);
};

//-----------------------------------------------------------------------------
template <typename FP> static constexpr FPType get_fp_type() {
  if constexpr (cpp::is_same_v<FP, float> && __FLT_MANT_DIG__ == 24)
    return FPType::IEEE754_Binary32;
  else if constexpr (cpp::is_same_v<FP, double> && __DBL_MANT_DIG__ == 53)
    return FPType::IEEE754_Binary64;
  else if constexpr (cpp::is_same_v<FP, long double>) {
    if constexpr (__LDBL_MANT_DIG__ == 53)
      return FPType::IEEE754_Binary64;
    else if constexpr (__LDBL_MANT_DIG__ == 64)
      return FPType::X86_Binary80;
    else if constexpr (__LDBL_MANT_DIG__ == 113)
      return FPType::IEEE754_Binary128;
  }
#if defined(LIBC_COMPILER_HAS_C23_FLOAT16)
  else if constexpr (cpp::is_same_v<FP, _Float16>)
    return FPType::IEEE754_Binary16;
#endif
#if defined(LIBC_COMPILER_HAS_C23_FLOAT128)
  else if constexpr (cpp::is_same_v<FP, _Float128>)
    return FPType::IEEE754_Binary128;
#endif
#if defined(LIBC_COMPILER_HAS_FLOAT128_EXTENSION)
  else if constexpr (cpp::is_same_v<FP, __float128>)
    return FPType::IEEE754_Binary128;
#endif
  else
    static_assert(cpp::always_false<FP>, "Unsupported type");
}

template <typename FP>
struct FloatProperties : public FPProperties<get_fp_type<FP>()> {};

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOATPROPERTIES_H
