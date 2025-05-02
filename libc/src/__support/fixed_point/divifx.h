//===-- Division of integers by fixed-point numbers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides implementations for functions that divide a standard
// integer type by a fixed-point type, returning a standard integer type result.
// This corresponds to the divi<fx> family (e.g., divir, divik) described
// in ISO/IEC TR 18037:2008 Annex C.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_INT_DIV_FX_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_INT_DIV_FX_H

#include "include/llvm-libc-macros/stdfix-macros.h" // Fixed-point types
#include "src/__support/CPP/bit.h"                  // bit_cast
#include "src/__support/CPP/limits.h"               // numeric_limits (optional)
#include "src/__support/CPP/type_traits.h"          // conditional_t, is_same_v
#include "src/__support/macros/attributes.h"        // LIBC_INLINE
#include "src/__support/macros/config.h" // LIBC_NAMESPACE_DECL, LIBC_COMPILER_HAS_FIXED_POINT
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "fx_rep.h" // FXRep for type info (FRACTION_LEN, StorageType)

// Only define contents if the compiler supports fixed-point types
#ifdef LIBC_COMPILER_HAS_FIXED_POINT

// Check for 128-bit integer support needed for high precision intermediates
#if defined(__SIZEOF_INT128__)
#define LIBC_INTERNAL_HAS_INT128
using int128_t = __int128_t;
using uint128_t = __uint128_t;
#endif

namespace LIBC_NAMESPACE_DECL {
namespace fixed_point {
namespace internal {

// --- Helper type traits for selecting intermediate calculation types ---

template <typename IntType, int FractionalBits>
using SelectDivIntermediateSigned =
    cpp::conditional_t<
        (sizeof(IntType) * 8 - 1 + FractionalBits <= 64), int64_t,
#ifdef LIBC_INTERNAL_HAS_INT128
        cpp::conditional_t<(sizeof(IntType) * 8 - 1 + FractionalBits <= 128),
                           int128_t,
                           void>
#else
        void
#endif
        >;

template <typename IntType, int FractionalBits>
using SelectDivIntermediateUnsigned = cpp::conditional_t<
    (sizeof(IntType) * 8 - 1 + FractionalBits <= 64), uint64_t,
#ifdef LIBC_INTERNAL_HAS_INT128
    cpp::conditional_t<(sizeof(IntType) * 8 - 1 + FractionalBits <= 128),
                       uint128_t, void>
#else
    void
#endif
    >;

// --- Core implementation template ---


template <typename IntType, typename FxType>
LIBC_INLINE IntType divifx_impl(IntType i, FxType fx) {
  // Get metadata about the fixed-point type using FXRep helper
  using FX = FXRep<FxType>;
  using StorageType = typename FX::StorageType;
  constexpr int F = FX::FRACTION_LEN;        // Number of fractional bits
  constexpr bool FxIsSigned = FX::SIGN_LEN; // Is the fx type signed?

  // Extract the raw integer bits from the fixed-point divisor
  StorageType raw_fx = cpp::bit_cast<StorageType>(fx);

  volatile StorageType check_raw_fx = raw_fx;
  if (LIBC_UNLIKELY(check_raw_fx == 0)) {

  }

  // Select appropriately sized intermediate types for the calculation
  using IntermediateSigned = SelectDivIntermediateSigned<IntType, F>;
  using IntermediateUnsigned = SelectDivIntermediateUnsigned<IntType, F>;

  // Compile-time check: ensure a wide enough type was found.
  static_assert(!cpp::is_same_v<IntermediateSigned, void>,
                "Calculation requires intermediate precision exceeding "
                "available types (int64_t or __int128_t).");

  // Calculate the numerator: (i << F)
  // Use the signed intermediate type for the numerator.
  IntermediateSigned num = static_cast<IntermediateSigned>(i) << F;

  // Perform the division: num / raw_fx
  IntermediateSigned intermediate_result;
  if constexpr (FxIsSigned) {
    IntermediateSigned den = static_cast<IntermediateSigned>(raw_fx);
    intermediate_result = num / den;
  } else {
    IntermediateUnsigned den = static_cast<IntermediateUnsigned>(raw_fx);
    intermediate_result = num / den;
  }

  return static_cast<IntType>(intermediate_result);
}

} // namespace internal

//===----------------------------------------------------------------------===//
// Public API: divi<fx> functions
//===----------------------------------------------------------------------===//

// --- Signed Fract Types ---

/** Divides int by fract, returns int. */
LIBC_INLINE int divir(int i, fract f) {
  return internal::divifx_impl<int, fract>(i, f);
}

} // namespace fixed_point
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_INT_DIV_FX_H
