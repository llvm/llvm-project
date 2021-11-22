//===-- Optimizer/Support/TypeCode.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_TYPECODE_H
#define FORTRAN_OPTIMIZER_SUPPORT_TYPECODE_H

#include "flang/ISO_Fortran_binding.h"
#include "llvm/Support/ErrorHandling.h"

namespace fir {

//===----------------------------------------------------------------------===//
// Translations of category and bitwidths to the type codes defined in flang's
// ISO_Fortran_binding.h.
//===----------------------------------------------------------------------===//

inline int characterBitsToTypeCode(unsigned bitwidth) {
  // clang-format off
  switch (bitwidth) {
  case 8:  return CFI_type_char;
  case 16: return CFI_type_char16_t;
  case 32: return CFI_type_char32_t;
  default: llvm_unreachable("unsupported character size");
  }
  // clang-format on
}

inline int complexBitsToTypeCode(unsigned bitwidth) {
  // clang-format off
  switch (bitwidth) {
  case 32:  return CFI_type_float_Complex;
  case 64:  return CFI_type_double_Complex;
  case 80:
  case 128: return CFI_type_long_double_Complex;
  default:  llvm_unreachable("unsupported complex size");
  }
  // clang-format on
}

inline int integerBitsToTypeCode(unsigned bitwidth) {
  // clang-format off
  switch (bitwidth) {
  case 8:   return CFI_type_int8_t;
  case 16:  return CFI_type_int16_t;
  case 32:  return CFI_type_int32_t;
  case 64:  return CFI_type_int64_t;
  case 128: return CFI_type_int128_t;
  default:  llvm_unreachable("unsupported integer size");
  }
  // clang-format on
}

inline int logicalBitsToTypeCode(unsigned bitwidth) {
  // clang-format off
  switch (bitwidth) {
  case 8: return CFI_type_Bool;
  case 16: return CFI_type_int_least16_t;
  case 32: return CFI_type_int_least32_t;
  case 64: return CFI_type_int_least64_t;
  default: llvm_unreachable("unsupported logical size");
  }
  // clang-format on
}

inline int realBitsToTypeCode(unsigned bitwidth) {
  // clang-format off
  switch (bitwidth) {
  case 32:  return CFI_type_float;
  case 64:  return CFI_type_double;
  case 80:
  case 128: return CFI_type_long_double;
  default:  llvm_unreachable("unsupported real size");
  }
  // clang-format on
}

static constexpr int derivedToTypeCode() { return CFI_type_struct; }

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_TYPECODE_H
