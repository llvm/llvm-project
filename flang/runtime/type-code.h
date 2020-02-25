//===-- runtime/type-code.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TYPE_CODE_H_
#define FORTRAN_RUNTIME_TYPE_CODE_H_

#include "flang/ISO_Fortran_binding.h"
#include "flang/Common/Fortran.h"

namespace Fortran::runtime {

using common::TypeCategory;

class TypeCode {
public:
  TypeCode() {}
  explicit TypeCode(ISO::CFI_type_t t) : raw_{t} {}
  TypeCode(TypeCategory, int);

  int raw() const { return raw_; }

  constexpr bool IsValid() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_struct;
  }
  constexpr bool IsInteger() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_ptrdiff_t;
  }
  constexpr bool IsReal() const {
    return raw_ >= CFI_type_float && raw_ <= CFI_type_long_double;
  }
  constexpr bool IsComplex() const {
    return raw_ >= CFI_type_float_Complex &&
        raw_ <= CFI_type_long_double_Complex;
  }
  constexpr bool IsCharacter() const { return raw_ == CFI_type_char; }
  constexpr bool IsLogical() const { return raw_ == CFI_type_Bool; }
  constexpr bool IsDerived() const { return raw_ == CFI_type_struct; }

  constexpr bool IsIntrinsic() const { return IsValid() && !IsDerived(); }

  constexpr TypeCategory Categorize() const {
    if (IsInteger()) {
      return TypeCategory::Integer;
    }
    if (IsReal()) {
      return TypeCategory::Real;
    }
    if (IsComplex()) {
      return TypeCategory::Complex;
    }
    if (IsCharacter()) {
      return TypeCategory::Character;
    }
    if (IsLogical()) {
      return TypeCategory::Logical;
    }
    return TypeCategory::Derived;
  }

private:
  ISO::CFI_type_t raw_{CFI_type_other};
};
}
#endif  // FORTRAN_RUNTIME_TYPE_CODE_H_
