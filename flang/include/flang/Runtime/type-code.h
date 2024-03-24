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
#include "flang/Common/optional.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
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

  constexpr RT_API_ATTRS bool IsValid() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_TYPE_LAST;
  }
  constexpr RT_API_ATTRS bool IsInteger() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_ptrdiff_t;
  }
  constexpr RT_API_ATTRS bool IsReal() const {
    return raw_ >= CFI_type_half_float && raw_ <= CFI_type_float128;
  }
  constexpr RT_API_ATTRS bool IsComplex() const {
    return raw_ >= CFI_type_half_float_Complex &&
        raw_ <= CFI_type_float128_Complex;
  }
  constexpr RT_API_ATTRS bool IsCharacter() const {
    return raw_ == CFI_type_char || raw_ == CFI_type_char16_t ||
        raw_ == CFI_type_char32_t;
  }
  constexpr RT_API_ATTRS bool IsLogical() const {
    return raw_ == CFI_type_Bool ||
        (raw_ >= CFI_type_int_least8_t && raw_ <= CFI_type_int_least64_t);
  }
  constexpr RT_API_ATTRS bool IsDerived() const {
    return raw_ == CFI_type_struct;
  }
  constexpr RT_API_ATTRS bool IsIntrinsic() const {
    return IsValid() && !IsDerived();
  }

  RT_API_ATTRS Fortran::common::optional<std::pair<TypeCategory, int>>
  GetCategoryAndKind() const;

  RT_API_ATTRS bool operator==(TypeCode that) const {
    if (raw_ == that.raw_) { // fast path
      return true;
    } else {
      // Multiple raw CFI_type_... codes can represent the same Fortran
      // type category + kind type parameter, e.g. CFI_type_int and
      // CFI_type_int32_t.
      auto thisCK{GetCategoryAndKind()};
      auto thatCK{that.GetCategoryAndKind()};
      return thisCK && thatCK && *thisCK == *thatCK;
    }
  }
  RT_API_ATTRS bool operator!=(TypeCode that) const { return !(*this == that); }

private:
  ISO::CFI_type_t raw_{CFI_type_other};
};
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_TYPE_CODE_H_
