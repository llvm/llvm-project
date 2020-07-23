//===-- Lower/Support/TypeCode.h --------------------------------*- C++ -*-===//
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

#ifndef LOWER_SUPPORT_TYPECODE_H
#define LOWER_SUPPORT_TYPECODE_H

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

// FIXME: LOGICAL has no type codes defined; using integer for now
inline int logicalBitsToTypeCode(unsigned bits) {
  llvm_unreachable("logical type has no direct support; use integer");
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

} // namespace fir

#endif // LOWER_SUPPORT_TYPECODE_H
