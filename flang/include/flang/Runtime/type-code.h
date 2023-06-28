//===-- include/flang/Runtime/type-code.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TYPE_CODE_H_
#define FORTRAN_RUNTIME_TYPE_CODE_H_

#include "flang/Common/Fortran.h"
#include "flang/ISO_Fortran_binding.h"
#include <optional>
#include <utility>

namespace Fortran::runtime {

using common::TypeCategory;

class TypeCode {
public:
  TypeCode() {}
  explicit RT_API_ATTRS TypeCode(ISO::CFI_type_t t) : raw_{t} {}
  RT_API_ATTRS TypeCode(TypeCategory, int kind);

  RT_API_ATTRS int raw() const { return raw_; }

  constexpr bool IsValid() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_TYPE_LAST;
  }
  constexpr bool IsInteger() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_ptrdiff_t;
  }
  constexpr bool IsReal() const {
    return raw_ >= CFI_type_half_float && raw_ <= CFI_type_float128;
  }
  constexpr bool IsComplex() const {
    return raw_ >= CFI_type_half_float_Complex &&
        raw_ <= CFI_type_float128_Complex;
  }
  constexpr bool IsCharacter() const {
    return raw_ == CFI_type_char || raw_ == CFI_type_char16_t ||
        raw_ == CFI_type_char32_t;
  }
  constexpr bool IsLogical() const {
    return raw_ == CFI_type_Bool ||
        (raw_ >= CFI_type_int_least8_t && raw_ <= CFI_type_int_least64_t);
  }
  constexpr bool IsDerived() const { return raw_ == CFI_type_struct; }
  constexpr bool IsIntrinsic() const { return IsValid() && !IsDerived(); }

  RT_API_ATTRS std::optional<std::pair<TypeCategory, int>>
  GetCategoryAndKind() const;

  RT_API_ATTRS bool operator==(const TypeCode &that) const {
    return raw_ == that.raw_;
  }
  bool operator!=(const TypeCode &that) const { return raw_ != that.raw_; }

private:
  ISO::CFI_type_t raw_{CFI_type_other};
};
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_TYPE_CODE_H_
