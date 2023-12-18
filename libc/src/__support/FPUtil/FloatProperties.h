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

// Returns the 'FPType' associated with this 'FP' type.
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
  using StorageType = uint16_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 16;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 10;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 5;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary32> {
  using StorageType = uint32_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 32;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 23;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 8;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary64> {
  using StorageType = uint64_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 64;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 52;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 11;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary128> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 128;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 112;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::X86_Binary80> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 80;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 64;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr auto ENCODING =
      FPEncoding::X86_ExtendedPrecision;
};

} // namespace internal

template <FPType fp_type>
struct FPProperties : public internal::FPBaseProperties<fp_type> {
private:
  using UP = internal::FPBaseProperties<fp_type>;

public:
  //---------------------------------------------------------------------------
  // Physical layer
  //
  // At this level, we only describe the bit *ranges* not necessarily their
  // meaning. A *floating point number* is composed of three parts : a `sign`,
  // an `exponent` and a `significand`. By convention, they are also designated
  // by a single letter : `s`, `e` and `m`. Note that `m` here stands for
  // *mantissa* but its use is generally discouraged :
  // https://en.wikipedia.org/wiki/Significand#Terminology
  //
  // For virtually all of the floating point number formats (and for the purpose
  // of this library) the bit ranges are in this exact order : `sign`,
  // `exponent` and `significand`.
  //
  // e.g. Depiction of 'IEEE754 Float16'
  //
  //      | sign   exponent          significand
  //      |    |  ┌───────┐  ┌─────────────────┐
  //      |    0  0 1 1 0 0  0 1 0 0 0 0 0 0 0 0
  //  bit |   15 14      10  9                 0
  //
  //---------------------------------------------------------------------------
  // Types
  //---------------------------------------------------------------------------
  // An unsigned integer that is wide enough to contain all of the floating
  // point bits.
  using StorageType = typename UP::StorageType;

  //---------------------------------------------------------------------------
  // Properties
  //---------------------------------------------------------------------------

  // The number of bits in StorageType.
  LIBC_INLINE_VAR static constexpr int STORAGE_LEN =
      sizeof(StorageType) * CHAR_BIT;

  // The number of bits to represent sign. For documentation purpose, always 1.
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  using UP::EXP_LEN;   // The number of bits for the *exponent* part
  using UP::SIG_LEN;   // The number of bits for the *significand* part
  using UP::TOTAL_LEN; // For convenience, the sum of `SIG_LEN`, `EXP_LEN`,
                       // and `SIGN_LEN`.
  static_assert(SIGN_LEN + EXP_LEN + SIG_LEN == TOTAL_LEN);
  static_assert(STORAGE_LEN >= TOTAL_LEN);

private:
  // Helper to set a single bit at 'position'.
  LIBC_INLINE static constexpr StorageType bit_at(int position) {
    return StorageType(1) << position;
  }

  // The shift amount to get the *significand* part to the least significant
  // bit. Always `0` but kept for consistency.
  LIBC_INLINE_VAR static constexpr int SIG_MASK_SHIFT = 0;
  // The shift amount to get the *exponent* part to the least significant bit.
  LIBC_INLINE_VAR static constexpr int EXP_MASK_SHIFT = SIG_LEN;
  // The shift amount to get the *sign* part to the least significant bit.
  LIBC_INLINE_VAR static constexpr int SIGN_MASK_SHIFT = SIG_LEN + EXP_LEN;

public:
  // The bit pattern that keeps only the *significand* part.
  LIBC_INLINE_VAR static constexpr StorageType SIG_MASK =
      mask_trailing_ones<StorageType, SIG_LEN>() << SIG_MASK_SHIFT;
  // The bit pattern that keeps only the *exponent* part.
  LIBC_INLINE_VAR static constexpr StorageType EXP_MASK =
      mask_trailing_ones<StorageType, EXP_LEN>() << EXP_MASK_SHIFT;
  // The bit pattern that keeps only the *sign* part.
  LIBC_INLINE_VAR static constexpr StorageType SIGN_MASK =
      mask_trailing_ones<StorageType, SIGN_LEN>() << SIGN_MASK_SHIFT;
  // The bit pattern that keeps only the *exponent + significand* part.
  LIBC_INLINE_VAR static constexpr StorageType EXP_SIG_MASK =
      mask_trailing_ones<StorageType, EXP_LEN + SIG_LEN>();
  // The bit pattern that keeps only the *sign + exponent + significand* part.
  LIBC_INLINE_VAR static constexpr StorageType FP_MASK =
      mask_trailing_ones<StorageType, TOTAL_LEN>();

  static_assert((SIG_MASK & EXP_MASK & SIGN_MASK) == 0, "masks disjoint");
  static_assert((SIG_MASK | EXP_MASK | SIGN_MASK) == FP_MASK, "masks cover");

  // The exponent bias. Always positive.
  LIBC_INLINE_VAR static constexpr int32_t EXP_BIAS =
      (1U << (EXP_LEN - 1U)) - 1U;
  static_assert(EXP_BIAS > 0);

  //---------------------------------------------------------------------------
  // Semantic layer
  //
  // At this level, we define properties and methods of the floating point
  // number - regardless of how it's encoded. We typically find methods to
  // extract the 'sign', 'exponent' and 'significand' of the number or whether
  // it represents peculiar values like 'infinity' or 'NaN'.
  //
  // A central notion when interacting with the physical layer is the concept of
  // 'normalized number'. A normalized number is the association of an exponent
  // and a significand in a _normalized form_, that is,  when its leading `1` is
  // at the position of the most significant bit.
  //
  // e.g., Here is the `1.01` normalized significand stored in an `uint16_t`.
  // The leading '1' is at the most significant bit position.
  //
  //    ┌──────────────────────────────┐
  //     1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
  //    15                             0
  //
  // Note that it is always possible to normalize a (non-zero) number by
  // decreasing its associated exponent. For instance, the value `0.001p0` can
  // be rewritten `1p-3`. Likewise `100p0` can be rewritten `1p2`. It becomes
  // obvious that the same number can be represented in a number of ways, hence
  // the importance of a normalized representation.
  //
  // Because 'significand' and 'exponent' usually need to be processed together
  // they are paired together in the 'Number' type below. This type

  //---------------------------------------------------------------------------
  // Type
  //---------------------------------------------------------------------------

  // This is a working format to represent an unsigned floating point number.
  // Note that numbers expressed in this format have a greater precision than
  // the physical format:
  // - The exponent is always 32-bit, this is always greater than 'EXP_LEN' (at
  //   most 15-bit).
  // - The significand is 'STORAGE_LEN'-bit, this is greater than 'SIG_LEN' by
  //   'EXP_LEN' bits for IEEE754 format, and by `EXP_LEN + 1` bits for
  //   X86_ExtendedPrecision.
  struct Number {
    StorageType significant = 0;
    int32_t exponent = 0;

    // Returns this Number in a normalized form, i.e., if non-zero, the leading
    // one is at the most significant bit position.
    Number normalize() const;
  };

  //---------------------------------------------------------------------------
  // Properties
  //---------------------------------------------------------------------------

private:
  LIBC_INLINE_VAR static constexpr StorageType QNAN_MASK =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision
          ? bit_at(SIG_LEN - 1) | bit_at(SIG_LEN - 2) // 0b1100...
          : bit_at(SIG_LEN - 1);                      // 0b1000...

  LIBC_INLINE_VAR static constexpr StorageType SNAN_MASK =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision
          ? bit_at(SIG_LEN - 1) | bit_at(SIG_LEN - 3) // 0b1010...
          : bit_at(SIG_LEN - 2);                      // 0b0100...

public:
  // The number of bits after the decimal dot when the number is in normal form.
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision ? SIG_LEN - 1
                                                                  : SIG_LEN;
  LIBC_INLINE_VAR static constexpr uint32_t MANTISSA_PRECISION =
      FRACTION_LEN + 1;
  LIBC_INLINE_VAR static constexpr StorageType FRACTION_MASK =
      mask_trailing_ones<StorageType, FRACTION_LEN>();

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr StorageType QUIET_NAN_MASK = QNAN_MASK;

  //---------------------------------------------------------------------------
  // Observers
  //---------------------------------------------------------------------------

  // Returns whether the 'value' represents a negative number.
  LIBC_INLINE static constexpr bool sign(StorageType value) {
    return value & SIGN_MASK;
  }

  //---------------------------------------------------------------------------
  // Modifiers
  //---------------------------------------------------------------------------

  // Returns the absolute value of 'value'.
  LIBC_INLINE static constexpr StorageType abs(StorageType value) {
    return value & EXP_SIG_MASK;
  }
};

template <typename FP>
struct FloatProperties : public FPProperties<get_fp_type<FP>()> {};

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOATPROPERTIES_H
