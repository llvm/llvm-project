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
#include "src/__support/macros/attributes.h" // LIBC_INLINE, LIBC_INLINE_VAR
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

// For now 'FPEncoding', 'FPBaseProperties' and 'FPCommonProperties' are
// implementation details.
namespace internal {

// The type of encoding for supported floating point types.
enum class FPEncoding {
  IEEE754,
  X86_ExtendedPrecision,
};

template <FPType> struct FPBaseProperties {};

template <> struct FPBaseProperties<FPType::IEEE754_Binary16> {
  using UIntType = uint16_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_BITS = 16;
  LIBC_INLINE_VAR static constexpr int SIG_BITS = 10;
  LIBC_INLINE_VAR static constexpr int EXP_BITS = 5;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary32> {
  using UIntType = uint32_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_BITS = 32;
  LIBC_INLINE_VAR static constexpr int SIG_BITS = 23;
  LIBC_INLINE_VAR static constexpr int EXP_BITS = 8;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary64> {
  using UIntType = uint64_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_BITS = 64;
  LIBC_INLINE_VAR static constexpr int SIG_BITS = 52;
  LIBC_INLINE_VAR static constexpr int EXP_BITS = 11;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary128> {
  using UIntType = UInt128;
  LIBC_INLINE_VAR static constexpr int TOTAL_BITS = 128;
  LIBC_INLINE_VAR static constexpr int SIG_BITS = 112;
  LIBC_INLINE_VAR static constexpr int EXP_BITS = 15;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::X86_Binary80> {
  using UIntType = UInt128;
  LIBC_INLINE_VAR static constexpr int TOTAL_BITS = 80;
  LIBC_INLINE_VAR static constexpr int SIG_BITS = 64;
  LIBC_INLINE_VAR static constexpr int EXP_BITS = 15;
  LIBC_INLINE_VAR static constexpr auto ENCODING =
      FPEncoding::X86_ExtendedPrecision;
};

// Derives more properties from 'FPBaseProperties' above.
// This class serves as a halfway point between 'FPBaseProperties' and
// 'FPProperties' below.
template <FPType fp_type>
struct FPCommonProperties : private FPBaseProperties<fp_type> {
  using UP = FPBaseProperties<fp_type>;
  using BitsType = typename UP::UIntType;

  LIBC_INLINE_VAR static constexpr uint32_t BIT_WIDTH = UP::TOTAL_BITS;
  LIBC_INLINE_VAR static constexpr uint32_t MANTISSA_WIDTH = UP::SIG_BITS;
  LIBC_INLINE_VAR static constexpr uint32_t EXPONENT_WIDTH = UP::EXP_BITS;

  // The exponent bias. Always positive.
  LIBC_INLINE_VAR static constexpr uint32_t EXPONENT_BIAS =
      (1U << (UP::EXP_BITS - 1U)) - 1U;
  static_assert(EXPONENT_BIAS > 0);
};

} // namespace internal

template <FPType> struct FPProperties {};

// ----------------
// Work In Progress
// ----------------
// The 'FPProperties' template specializations below are being slowly replaced
// with properties from 'FPCommonProperties' above. Once specializations are
// empty, 'FPProperties' declaration can be fully replace with
// 'FPCommonProperties' implementation.

template <>
struct FPProperties<FPType::IEEE754_Binary32>
    : public internal::FPCommonProperties<FPType::IEEE754_Binary32> {
  // The mantissa precision includes the implicit bit.
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  static constexpr BitsType SIGN_MASK = BitsType(1)
                                        << (EXPONENT_WIDTH + MANTISSA_WIDTH);
  static constexpr BitsType EXPONENT_MASK = ~(SIGN_MASK | MANTISSA_MASK);
  static constexpr BitsType EXP_MANT_MASK = MANTISSA_MASK + EXPONENT_MASK;
  static_assert(EXP_MANT_MASK == ~SIGN_MASK,
                "Exponent and mantissa masks are not as expected.");

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType QUIET_NAN_MASK = 0x00400000U;
};

template <>
struct FPProperties<FPType::IEEE754_Binary64>
    : public internal::FPCommonProperties<FPType::IEEE754_Binary64> {
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  static constexpr BitsType SIGN_MASK = BitsType(1)
                                        << (EXPONENT_WIDTH + MANTISSA_WIDTH);
  static constexpr BitsType EXPONENT_MASK = ~(SIGN_MASK | MANTISSA_MASK);
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
template <>
struct FPProperties<FPType::X86_Binary80>
    : public internal::FPCommonProperties<FPType::X86_Binary80> {
  static constexpr BitsType FULL_WIDTH_MASK = ((BitsType(1) << BIT_WIDTH) - 1);
  static constexpr uint32_t MANTISSA_WIDTH = 63;
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  // The x86 80 bit float represents the leading digit of the mantissa
  // explicitly. This is the mask for that bit.
  static constexpr BitsType EXPLICIT_BIT_MASK = (BitsType(1) << MANTISSA_WIDTH);
  static constexpr BitsType SIGN_MASK =
      BitsType(1) << (EXPONENT_WIDTH + MANTISSA_WIDTH + 1);
  static constexpr BitsType EXPONENT_MASK =
      ((BitsType(1) << EXPONENT_WIDTH) - 1) << (MANTISSA_WIDTH + 1);
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
template <>
struct FPProperties<FPType::IEEE754_Binary128>
    : public internal::FPCommonProperties<FPType::IEEE754_Binary128> {
  static constexpr uint32_t MANTISSA_PRECISION = MANTISSA_WIDTH + 1;
  static constexpr BitsType MANTISSA_MASK = (BitsType(1) << MANTISSA_WIDTH) - 1;
  static constexpr BitsType SIGN_MASK = BitsType(1)
                                        << (EXPONENT_WIDTH + MANTISSA_WIDTH);
  static constexpr BitsType EXPONENT_MASK = ~(SIGN_MASK | MANTISSA_MASK);
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
template <typename FP> LIBC_INLINE static constexpr FPType get_fp_type() {
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
