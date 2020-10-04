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
      raw_ = CFI_type_int_fast16_t;
      break;
    case 4:
      raw_ = CFI_type_int_fast32_t;
      break;
    case 8:
      raw_ = CFI_type_int_fast64_t;
      break;
    }
    break;
  case TypeCategory::Derived:
    raw_ = CFI_type_struct;
    break;
  }
}

std::optional<std::pair<TypeCategory, int>>
TypeCode::GetCategoryAndKind() const {
  switch (raw_) {
  case CFI_type_int8_t:
    return std::make_pair(TypeCategory::Integer, 1);
  case CFI_type_int16_t:
    return std::make_pair(TypeCategory::Integer, 2);
  case CFI_type_int32_t:
    return std::make_pair(TypeCategory::Integer, 4);
  case CFI_type_int64_t:
    return std::make_pair(TypeCategory::Integer, 8);
  case CFI_type_int128_t:
    return std::make_pair(TypeCategory::Integer, 16);
  case CFI_type_float:
    return std::make_pair(TypeCategory::Real, 4);
  case CFI_type_double:
    return std::make_pair(TypeCategory::Real, 8);
  case CFI_type_long_double:
#if __x86_64__
    return std::make_pair(TypeCategory::Real, 10);
#else
    return std::make_pair(TypeCategory::Real, 16);
#endif
  case CFI_type_float_Complex:
    return std::make_pair(TypeCategory::Complex, 4);
  case CFI_type_double_Complex:
    return std::make_pair(TypeCategory::Complex, 8);
  case CFI_type_long_double_Complex:
#if __x86_64__
    return std::make_pair(TypeCategory::Complex, 10);
#else
    return std::make_pair(TypeCategory::Complex, 16);
#endif
  case CFI_type_char:
    return std::make_pair(TypeCategory::Character, 1);
  case CFI_type_char16_t:
    return std::make_pair(TypeCategory::Character, 2);
  case CFI_type_char32_t:
    return std::make_pair(TypeCategory::Character, 4);
  case CFI_type_Bool:
    return std::make_pair(TypeCategory::Logical, 1);
  case CFI_type_int_fast8_t:
    return std::make_pair(TypeCategory::Logical, 1);
  case CFI_type_int_fast16_t:
    return std::make_pair(TypeCategory::Logical, 2);
  case CFI_type_int_fast32_t:
    return std::make_pair(TypeCategory::Logical, 4);
  case CFI_type_int_fast64_t:
    return std::make_pair(TypeCategory::Logical, 8);
  case CFI_type_struct:
    return std::make_pair(TypeCategory::Derived, 0);
  default:
    return std::nullopt;
  }
}
} // namespace Fortran::runtime
