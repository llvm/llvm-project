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

#ifndef OPTIMIZER_SUPPORT_TYPECODE_H
#define OPTIMIZER_SUPPORT_TYPECODE_H

#include "flang/ISO_Fortran_binding.h"
#include "llvm/Support/ErrorHandling.h"

namespace fir {

//===----------------------------------------------------------------------===//
// Translations of category and bitwidths to the type codes defined in flang's
// ISO_Fortran_binding.h.
//===----------------------------------------------------------------------===//

inline int characterBitsToTypeCode(unsigned bits) {
  // clang-format off
  switch (bits) {
  case 8:  return CFI_type_char;
  case 16: return CFI_type_char16_t;
  case 32: return CFI_type_char32_t;
  default: llvm_unreachable("unsupported character size");
  }
  // clang-format on
}

inline int complexBitsToTypeCode(unsigned bits) {
  // clang-format off
  switch (bits) {
  case 32:  return CFI_type_float_Complex;
  case 64:  return CFI_type_double_Complex;
  case 80:
  case 128: return CFI_type_long_double_Complex;
  default:  llvm_unreachable("unsupported complex size");
  }
  // clang-format on
}

inline int integerBitsToTypeCode(unsigned bits) {
  // clang-format off
  switch (bits) {
  case 8:   return CFI_type_int8_t;
  case 16:  return CFI_type_int16_t;
  case 32:  return CFI_type_int32_t;
  case 64:  return CFI_type_int64_t;
  case 128: return CFI_type_int128_t;
  default:  llvm_unreachable("unsupported integer size");
  }
  // clang-format on
}

// Always use CFI_type_Bool and let the rest get sorted out by the elem_size.
// NB: do *not* use the CFI_type_intN_t codes. The flang runtime will choke.
inline int logicalBitsToTypeCode(unsigned bits) {
  // clang-format off
  switch (bits) {
  case 8:
  case 16:
  case 32:
  case 64: return CFI_type_Bool;
  default: llvm_unreachable("unsupported logical size");
  }
  // clang-format on
}

inline int realBitsToTypeCode(unsigned bits) {
  // clang-format off
  switch (bits) {
  case 32:  return CFI_type_float;
  case 64:  return CFI_type_double;
  case 80:
  case 128: return CFI_type_long_double;
  default:  llvm_unreachable("unsupported real size");
  }
  // clang-format on
}

static constexpr int derivedToTypeCode() {
  return CFI_type_struct;
}

} // namespace fir

#endif // OPTIMIZER_SUPPORT_TYPECODE_H
