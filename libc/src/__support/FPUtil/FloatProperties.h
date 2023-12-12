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
#include "src/__support/math_extras.h"             // mask_trailing_ones

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

} // namespace internal

template <FPType fp_type>
struct FPProperties : public internal::FPBaseProperties<fp_type> {
private:
  using UP = internal::FPBaseProperties<fp_type>;
  using UP::EXP_BITS;
  using UP::SIG_BITS;
  using UP::TOTAL_BITS;
  using UIntType = typename UP::UIntType;

  LIBC_INLINE_VAR static constexpr int STORAGE_BITS =
      sizeof(UIntType) * CHAR_BIT;
  static_assert(STORAGE_BITS >= TOTAL_BITS);

  // The number of bits to represent sign.
  // For documentation purpose, always 1.
  LIBC_INLINE_VAR static constexpr int SIGN_BITS = 1;
  static_assert(SIGN_BITS + EXP_BITS + SIG_BITS == TOTAL_BITS);

  // The exponent bias. Always positive.
  LIBC_INLINE_VAR static constexpr int32_t EXP_BIAS =
      (1U << (EXP_BITS - 1U)) - 1U;
  static_assert(EXP_BIAS > 0);

  // Shifts
  LIBC_INLINE_VAR static constexpr int SIG_MASK_SHIFT = 0;
  LIBC_INLINE_VAR static constexpr int EXP_MASK_SHIFT = SIG_BITS;
  LIBC_INLINE_VAR static constexpr int SIGN_MASK_SHIFT = SIG_BITS + EXP_BITS;

  // Masks
  LIBC_INLINE_VAR static constexpr UIntType SIG_MASK =
      mask_trailing_ones<UIntType, SIG_BITS>() << SIG_MASK_SHIFT;
  LIBC_INLINE_VAR static constexpr UIntType EXP_MASK =
      mask_trailing_ones<UIntType, EXP_BITS>() << EXP_MASK_SHIFT;
  // Trailing underscore on SIGN_MASK_ is temporary - it will be removed
  // once we can replace the public part below with the private one.
  LIBC_INLINE_VAR static constexpr UIntType SIGN_MASK_ =
      mask_trailing_ones<UIntType, SIGN_BITS>() << SIGN_MASK_SHIFT;
  LIBC_INLINE_VAR static constexpr UIntType FP_MASK =
      mask_trailing_ones<UIntType, TOTAL_BITS>();
  static_assert((SIG_MASK & EXP_MASK & SIGN_MASK_) == 0, "masks disjoint");
  static_assert((SIG_MASK | EXP_MASK | SIGN_MASK_) == FP_MASK, "masks cover");

  LIBC_INLINE static constexpr UIntType bit_at(int position) {
    return UIntType(1) << position;
  }

  LIBC_INLINE_VAR static constexpr UIntType QNAN_MASK =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision
          ? bit_at(SIG_BITS - 1) | bit_at(SIG_BITS - 2) // 0b1100...
          : bit_at(SIG_BITS - 1);                       // 0b1000...

  LIBC_INLINE_VAR static constexpr UIntType SNAN_MASK =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision
          ? bit_at(SIG_BITS - 1) | bit_at(SIG_BITS - 3) // 0b1010...
          : bit_at(SIG_BITS - 2);                       // 0b0100...

  // The number of bits after the decimal dot when the number if in normal form.
  LIBC_INLINE_VAR static constexpr int FRACTION_BITS =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision ? SIG_BITS - 1
                                                                  : SIG_BITS;

public:
  // Public facing API to keep the change local to this file.
  using BitsType = UIntType;

  LIBC_INLINE_VAR static constexpr uint32_t BIT_WIDTH = TOTAL_BITS;
  LIBC_INLINE_VAR static constexpr uint32_t MANTISSA_WIDTH = FRACTION_BITS;
  LIBC_INLINE_VAR static constexpr uint32_t MANTISSA_PRECISION =
      MANTISSA_WIDTH + 1;
  LIBC_INLINE_VAR static constexpr BitsType MANTISSA_MASK =
      mask_trailing_ones<UIntType, MANTISSA_WIDTH>();
  LIBC_INLINE_VAR static constexpr uint32_t EXPONENT_WIDTH = EXP_BITS;
  LIBC_INLINE_VAR static constexpr int32_t EXPONENT_BIAS = EXP_BIAS;
  LIBC_INLINE_VAR static constexpr BitsType SIGN_MASK = SIGN_MASK_;
  LIBC_INLINE_VAR static constexpr BitsType EXPONENT_MASK = EXP_MASK;
  LIBC_INLINE_VAR static constexpr BitsType EXP_MANT_MASK = EXP_MASK | SIG_MASK;

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType QUIET_NAN_MASK = QNAN_MASK;
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
