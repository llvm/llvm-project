//===-- runtime/type-code.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/type-code.h"

namespace Fortran::runtime {

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS TypeCode::TypeCode(TypeCategory f, int kind) {
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
  case TypeCategory::Unsigned:
    switch (kind) {
    case 1:
      raw_ = CFI_type_uint8_t;
      break;
    case 2:
      raw_ = CFI_type_uint16_t;
      break;
    case 4:
      raw_ = CFI_type_uint32_t;
      break;
    case 8:
      raw_ = CFI_type_uint64_t;
      break;
    case 16:
      raw_ = CFI_type_uint128_t;
      break;
    }
    break;
  case TypeCategory::Real:
    switch (kind) {
    case 2:
      raw_ = CFI_type_half_float;
      break;
    case 3:
      raw_ = CFI_type_bfloat;
      break;
    case 4:
      raw_ = CFI_type_float;
      break;
    case 8:
      raw_ = CFI_type_double;
      break;
    case 10:
      raw_ = CFI_type_extended_double;
      break;
    case 16:
      raw_ = CFI_type_float128;
      break;
    }
    break;
  case TypeCategory::Complex:
    switch (kind) {
    case 2:
      raw_ = CFI_type_half_float_Complex;
      break;
    case 3:
      raw_ = CFI_type_bfloat_Complex;
      break;
    case 4:
      raw_ = CFI_type_float_Complex;
      break;
    case 8:
      raw_ = CFI_type_double_Complex;
      break;
    case 10:
      raw_ = CFI_type_extended_double_Complex;
      break;
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
      raw_ = CFI_type_int_least16_t;
      break;
    case 4:
      raw_ = CFI_type_int_least32_t;
      break;
    case 8:
      raw_ = CFI_type_int_least64_t;
      break;
    }
    break;
  case TypeCategory::Derived:
    raw_ = CFI_type_struct;
    break;
  }
}

RT_API_ATTRS Fortran::common::optional<std::pair<TypeCategory, int>>
TypeCode::GetCategoryAndKind() const {
  switch (raw_) {
  case CFI_type_signed_char:
    return std::make_pair(TypeCategory::Character, sizeof(signed char));
  case CFI_type_short:
    return std::make_pair(TypeCategory::Integer, sizeof(short));
  case CFI_type_int:
    return std::make_pair(TypeCategory::Integer, sizeof(int));
  case CFI_type_long:
    return std::make_pair(TypeCategory::Integer, sizeof(long));
  case CFI_type_long_long:
    return std::make_pair(TypeCategory::Integer, sizeof(long long));
  case CFI_type_size_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::size_t));
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
  case CFI_type_int_least8_t:
    return std::make_pair(TypeCategory::Logical, 1);
  case CFI_type_int_least16_t:
    return std::make_pair(TypeCategory::Logical, 2);
  case CFI_type_int_least32_t:
    return std::make_pair(TypeCategory::Logical, 4);
  case CFI_type_int_least64_t:
    return std::make_pair(TypeCategory::Logical, 8);
  case CFI_type_int_least128_t:
    return std::make_pair(TypeCategory::Integer, 16);
  case CFI_type_int_fast8_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::int_fast8_t));
  case CFI_type_int_fast16_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::int_fast16_t));
  case CFI_type_int_fast32_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::int_fast32_t));
  case CFI_type_int_fast64_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::int_fast64_t));
  case CFI_type_int_fast128_t:
    return std::make_pair(TypeCategory::Integer, 16);
  case CFI_type_intmax_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::intmax_t));
  case CFI_type_intptr_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::intptr_t));
  case CFI_type_ptrdiff_t:
    return std::make_pair(TypeCategory::Integer, sizeof(std::ptrdiff_t));
  case CFI_type_half_float:
    return std::make_pair(TypeCategory::Real, 2);
  case CFI_type_bfloat:
    return std::make_pair(TypeCategory::Real, 3);
  case CFI_type_float:
    return std::make_pair(TypeCategory::Real, 4);
  case CFI_type_double:
    return std::make_pair(TypeCategory::Real, 8);
  case CFI_type_extended_double:
    return std::make_pair(TypeCategory::Real, 10);
  case CFI_type_long_double:
    return std::make_pair(TypeCategory::Real, 16);
  case CFI_type_float128:
    return std::make_pair(TypeCategory::Real, 16);
  case CFI_type_half_float_Complex:
    return std::make_pair(TypeCategory::Complex, 2);
  case CFI_type_bfloat_Complex:
    return std::make_pair(TypeCategory::Complex, 3);
  case CFI_type_float_Complex:
    return std::make_pair(TypeCategory::Complex, 4);
  case CFI_type_double_Complex:
    return std::make_pair(TypeCategory::Complex, 8);
  case CFI_type_extended_double_Complex:
    return std::make_pair(TypeCategory::Complex, 10);
  case CFI_type_long_double_Complex:
    return std::make_pair(TypeCategory::Complex, 16);
  case CFI_type_float128_Complex:
    return std::make_pair(TypeCategory::Complex, 16);
  case CFI_type_Bool:
    return std::make_pair(TypeCategory::Logical, 1);
  case CFI_type_char:
    return std::make_pair(TypeCategory::Character, 1);
  case CFI_type_cptr:
    return std::make_pair(TypeCategory::Integer, sizeof(void *));
  case CFI_type_struct:
    return std::make_pair(TypeCategory::Derived, 0);
  case CFI_type_char16_t:
    return std::make_pair(TypeCategory::Character, 2);
  case CFI_type_char32_t:
    return std::make_pair(TypeCategory::Character, 4);
  case CFI_type_uint8_t:
    return std::make_pair(TypeCategory::Unsigned, 1);
  case CFI_type_uint16_t:
    return std::make_pair(TypeCategory::Unsigned, 2);
  case CFI_type_uint32_t:
    return std::make_pair(TypeCategory::Unsigned, 4);
  case CFI_type_uint64_t:
    return std::make_pair(TypeCategory::Unsigned, 8);
  case CFI_type_uint128_t:
    return std::make_pair(TypeCategory::Unsigned, 16);
  default:
    return Fortran::common::nullopt;
  }
}

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
