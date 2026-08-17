//===-- include/flang/Evaluate/typekind-traits.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_TYPEKINDTRAITS_H_
#define FORTRAN_EVALUATE_TYPEKINDTRAITS_H_

#include "flang/Common/Fortran-consts.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/integer-value.h"
#include "flang/Evaluate/real-value.h"

namespace Fortran::evaluate::value {
class CharacterValue;
class IntegerValue;
class ComplexValue;
} // namespace Fortran::evaluate::value

namespace Fortran::evaluate {

template <common::TypeCategory CAT, int KIND> struct TypeKind;

template <> struct TypeKind<common::TypeCategory::Character, 1> {
  using CharT = char;
  using StringT = std::basic_string<CharT>;
  using Scalar = value::CharacterValue;
  static constexpr int kind{1};
};

template <> struct TypeKind<common::TypeCategory::Character, 2> {
  using CharT = char16_t;
  using StringT = std::basic_string<CharT>;
  using Scalar = value::CharacterValue;
  static constexpr int kind{2};
};

template <> struct TypeKind<common::TypeCategory::Character, 4> {
  using CharT = char32_t;
  using StringT = std::basic_string<CharT>;
  using Scalar = value::CharacterValue;
  static constexpr int kind{4};
};

template <int KIND> struct TypeKind<common::TypeCategory::Integer, KIND> {
  static constexpr int kind{KIND};
  static constexpr int bits{value::IntegerValue::bits(KIND)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = SignedT;
  using Scalar = value::IntegerValue;
};

template <int KIND> struct TypeKind<common::TypeCategory::Unsigned, KIND> {
  static constexpr int kind{KIND};
  static constexpr int bits{value::IntegerValue::bits(KIND)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = UnsignedT;
  using Scalar = value::IntegerValue;
};

namespace detail {
// Only REAL(4) and REAL(8) have a portable native host arithmetic type
// (float and double, respectively); every other kind maps to void.
template <int BITS> struct RealHostType {
  using type = void;
};
template <> struct RealHostType<32> {
  using type = float;
};
template <> struct RealHostType<64> {
  using type = double;
};
} // namespace detail

template <int KIND> struct TypeKind<common::TypeCategory::Real, KIND> {
  static constexpr int kind{KIND};
  static constexpr int bits{value::RealValue::bits(KIND)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = typename detail::RealHostType<bits>::type;
  using Scalar = value::RealValue;
};

template <int KIND> struct TypeKind<common::TypeCategory::Complex, KIND> {
  static constexpr int kind{KIND};
  using Scalar = value::ComplexValue;
};

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_TYPEKINDTRAITS_H_
