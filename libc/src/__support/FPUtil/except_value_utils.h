//===-- Common header for helpers to set exceptional values -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_EXCEPT_VALUE_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_EXCEPT_VALUE_UTILS_H

#include "FEnvImpl.h"
#include "FPBits.h"
#include "rounding_mode.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE {

namespace fputil {

// This file contains utility functions and classes to manage exceptional values
// when there are many of them.
//
// Example usage:
//
// Define list of exceptional inputs and outputs:
//   static constexpr int N = ...;  // Number of exceptional values.
//   static constexpr fputil::ExceptValues<UIntType, N> Excepts {
//     <list of input bits, output bits and offsets>
//   };
//
// Check for exceptional inputs:
//   if (auto r = Excepts.lookup(x_bits); LIBC_UNLIKELY(r.has_value()))
//     return r.value();

template <typename T, size_t N> struct ExceptValues {
  static_assert(cpp::is_floating_point_v<T>, "Must be a floating point type.");

  using UIntType = typename FPBits<T>::UIntType;

  struct Mapping {
    UIntType input;
    UIntType rnd_towardzero_result;
    UIntType rnd_upward_offset;
    UIntType rnd_downward_offset;
    UIntType rnd_tonearest_offset;
  };

  Mapping values[N];

  LIBC_INLINE constexpr cpp::optional<T> lookup(UIntType x_bits) const {
    for (size_t i = 0; i < N; ++i) {
      if (LIBC_UNLIKELY(x_bits == values[i].input)) {
        UIntType out_bits = values[i].rnd_towardzero_result;
        switch (fputil::quick_get_round()) {
        case FE_UPWARD:
          out_bits += values[i].rnd_upward_offset;
          break;
        case FE_DOWNWARD:
          out_bits += values[i].rnd_downward_offset;
          break;
        case FE_TONEAREST:
          out_bits += values[i].rnd_tonearest_offset;
          break;
        }
        return FPBits<T>(out_bits).get_val();
      }
    }
    return cpp::nullopt;
  }

  LIBC_INLINE constexpr cpp::optional<T> lookup_odd(UIntType x_abs,
                                                    bool sign) const {
    for (size_t i = 0; i < N; ++i) {
      if (LIBC_UNLIKELY(x_abs == values[i].input)) {
        UIntType out_bits = values[i].rnd_towardzero_result;
        switch (fputil::quick_get_round()) {
        case FE_UPWARD:
          out_bits += sign ? values[i].rnd_downward_offset
                           : values[i].rnd_upward_offset;
          break;
        case FE_DOWNWARD:
          out_bits += sign ? values[i].rnd_upward_offset
                           : values[i].rnd_downward_offset;
          break;
        case FE_TONEAREST:
          out_bits += values[i].rnd_tonearest_offset;
          break;
        }
        T result = FPBits<T>(out_bits).get_val();
        if (sign)
          result = -result;

        return result;
      }
    }
    return cpp::nullopt;
  }
};

// Helper functions to set results for exceptional cases.
template <typename T> LIBC_INLINE T round_result_slightly_down(T value_rn) {
  volatile T tmp = value_rn;
  constexpr T MIN_NORMAL = FPBits<T>::min_normal().get_val();
  tmp = tmp - MIN_NORMAL;
  return tmp;
}

template <typename T> LIBC_INLINE T round_result_slightly_up(T value_rn) {
  volatile T tmp = value_rn;
  const T MIN_NORMAL = FPBits<T>::min_normal().get_val();
  tmp = tmp + MIN_NORMAL;
  return tmp;
}

} // namespace fputil

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_EXCEPT_VALUE_UTILS_H
