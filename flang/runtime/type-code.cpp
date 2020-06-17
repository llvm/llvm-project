//===-- runtime/type-code.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "type-code.h"

namespace Fortran::runtime {

TypeCode::TypeCode(TypeCategory f, int kind) {
  switch (f) {
  case TypeCategory::Integer:
    switch (kind) {
    case 1:
      raw_ = CFI_type_int8_t;
      break;
    case 2:
      raw_ = CFI_type_int16_t;
      break;
    case 4:
      raw_ = CFI_type_int32_t;
      break;
    case 8:
      raw_ = CFI_type_int64_t;
      break;
    case 16:
      raw_ = CFI_type_int128_t;
      break;
    }
    break;
  case TypeCategory::Real:
    switch (kind) {
    case 4:
      raw_ = CFI_type_float;
      break;
    case 8:
      raw_ = CFI_type_double;
      break;
    case 10:
    case 16:
      raw_ = CFI_type_long_double;
      break;
    }
    break;
  case TypeCategory::Complex:
    switch (kind) {
    case 4:
      raw_ = CFI_type_float_Complex;
      break;
    case 8:
      raw_ = CFI_type_double_Complex;
      break;
    case 10:
    case 16:
      raw_ = CFI_type_long_double_Complex;
      break;
    }
    break;
  case TypeCategory::Character:
    switch (kind) {
    case 1:
      raw_ = CFI_type_char;
      break;
    case 2:
      raw_ = CFI_type_char16_t;
      break;
    case 4:
      raw_ = CFI_type_char32_t;
      break;
    }
    break;
  case TypeCategory::Logical:
    switch (kind) {
    case 1:
      raw_ = CFI_type_Bool;
      break;
    case 2:
      raw_ = CFI_type_int16_t;
      break;
    case 4:
      raw_ = CFI_type_int32_t;
      break;
    case 8:
      raw_ = CFI_type_int64_t;
      break;
    }
    break;
  case TypeCategory::Derived:
    raw_ = CFI_type_struct;
    break;
  }
}
} // namespace Fortran::runtime
