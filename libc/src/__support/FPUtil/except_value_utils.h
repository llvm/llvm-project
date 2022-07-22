//===-- Common header for helpers to set exceptional values -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_EXCEPT_VALUE_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_EXCEPT_VALUE_UTILS_H

#include "FEnvImpl.h"
#include "FPBits.h"

namespace __llvm_libc {

namespace fputil {

template <typename T, int N> struct ExceptionalValues {
  using UIntType = typename FPBits<T>::UIntType;
  static constexpr int SIZE = N;
  // Input bits.
  UIntType inputs[SIZE];
  // Output bits contains 4 values:
  //   output[i][0]: output bits corresponding to FE_TOWARDZERO
  //   output[i][1]: offset for FE_UPWARD
  //   output[i][2]: offset for FE_DOWNWARD
  //   output[i][3]: offset for FE_TONEAREST
  UIntType outputs[SIZE][4];
};

template <typename T, int N> struct ExceptionChecker {
  using UIntType = typename FPBits<T>::UIntType;
  using FPBits = FPBits<T>;
  using ExceptionalValues = ExceptionalValues<T, N>;

  static bool check_odd_func(const ExceptionalValues &ExceptVals,
                             UIntType x_abs, bool sign, T &result) {
    for (int i = 0; i < N; ++i) {
      if (unlikely(x_abs == ExceptVals.inputs[i])) {
        UIntType out_bits = ExceptVals.outputs[i][0]; // FE_TOWARDZERO
        switch (fputil::get_round()) {
        case FE_UPWARD:
          out_bits +=
              sign ? ExceptVals.outputs[i][2] : ExceptVals.outputs[i][1];
          break;
        case FE_DOWNWARD:
          out_bits +=
              sign ? ExceptVals.outputs[i][1] : ExceptVals.outputs[i][2];
          break;
        case FE_TONEAREST:
          out_bits += ExceptVals.outputs[i][3];
          break;
        }
        result = FPBits(out_bits).get_val();
        if (sign)
          result = -result;

        return true;
      }
    }
    return false;
  }
};

} // namespace fputil

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_EXCEPT_VALUE_UTILS_H
