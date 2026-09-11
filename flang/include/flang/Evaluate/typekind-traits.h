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
#include "flang/Evaluate/logical-value.h"
#include "flang/Evaluate/real-value.h"
#include "flang/Evaluate/type.h"

namespace Fortran::evaluate::value {
class CharacterValue;
class IntegerValue;
class ComplexValue;
} // namespace Fortran::evaluate::value

namespace Fortran::evaluate {

/// Traits class for Fortran intrinsics types.
///
/// In contrast to Type<CAT>, TypeKind<CAT,KIND> also carries the KIND as
/// template parameter. Used to resolve a Fortran type to an equivalent C/C++
/// type of the host compiler. Avoid using it anywhere else as template
/// instantiation for each KIND separately blows up build time.
///
/// Common members:
///  * category    The CAT template argument
///  * kind        The KIND template argument
///  * bits        For types where it makes sense, how many bits of information
///                it holds
///  * bytesStored How many bytes this type requires in memory; includes
///                alignment/padding bytes
///  * HostT       The equivalent C/C++ type in the host compiler; void if there
///                is no equivalent
///  * FortranType The equivalent evaluate::Type<CAT>
///  * GetType()   The equivalent evaluate::DynamicType
template <common::TypeCategory CAT, int KIND> struct TypeKind;

template <int KIND> struct TypeKind<common::TypeCategory::Integer, KIND> {
  static constexpr common::TypeCategory category{common::TypeCategory::Integer};
  static constexpr int kind{KIND};
  static constexpr int bits{value::IntegerValue::bits(KIND)};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = SignedT;
  using Scalar = value::IntegerValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Integer>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

using IntegerKindTypes = std::tuple<TypeKind<TypeCategory::Integer, 1>,
    TypeKind<TypeCategory::Integer, 2>, TypeKind<TypeCategory::Integer, 4>,
    TypeKind<TypeCategory::Integer, 8>, TypeKind<TypeCategory::Integer, 16>>;

template <int KIND> struct TypeKind<common::TypeCategory::Unsigned, KIND> {
  static constexpr common::TypeCategory category{
      common::TypeCategory::Unsigned};
  static constexpr int kind{KIND};
  static constexpr int bits{value::IntegerValue::bits(KIND)};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = UnsignedT;
  using Scalar = value::IntegerValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Unsigned>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

using UnsignedKindTypes = std::tuple<TypeKind<TypeCategory::Unsigned, 1>,
    TypeKind<TypeCategory::Unsigned, 2>, TypeKind<TypeCategory::Unsigned, 4>,
    TypeKind<TypeCategory::Unsigned, 8>, TypeKind<TypeCategory::Unsigned, 16>>;

template <int KIND> struct TypeKind<common::TypeCategory::Logical, KIND> {
  static constexpr common::TypeCategory category{common::TypeCategory::Logical};
  static constexpr int kind{KIND};
  static constexpr int bits{value::LogicalValue::bits(KIND)};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = UnsignedT;
  using Scalar = value::LogicalValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Logical>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

using LogicalKindTypes = std::tuple<TypeKind<TypeCategory::Logical, 1>,
    TypeKind<TypeCategory::Logical, 2>, TypeKind<TypeCategory::Logical, 4>>;

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
  static constexpr common::TypeCategory category{common::TypeCategory::Real};
  static constexpr int kind{KIND};
  static constexpr int bits{value::RealValue::bits(KIND)};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using UnsignedT = common::HostUnsignedIntType<bits>;
  using SignedT = common::HostSignedIntType<bits>;
  using HostT = typename detail::RealHostType<bits>::type;
  using Scalar = value::RealValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Real>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

using RealKindTypes =
    std::tuple<TypeKind<TypeCategory::Real, 2>, TypeKind<TypeCategory::Real, 3>,
        TypeKind<TypeCategory::Real, 4>, TypeKind<TypeCategory::Real, 8>,
        TypeKind<TypeCategory::Real, 10>, TypeKind<TypeCategory::Real, 16>>;

template <int KIND> struct TypeKind<common::TypeCategory::Complex, KIND> {
  static constexpr common::TypeCategory category{common::TypeCategory::Complex};
  static constexpr int kind{KIND};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Complex>;
  using Scalar = value::ComplexValue;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
  using Part = TypeKind<common::TypeCategory::Real, KIND>;
};

using ComplexKindTypes = std::tuple<TypeKind<TypeCategory::Complex, 2>,
    TypeKind<TypeCategory::Complex, 3>, TypeKind<TypeCategory::Complex, 4>,
    TypeKind<TypeCategory::Complex, 8>, TypeKind<TypeCategory::Complex, 10>,
    TypeKind<TypeCategory::Complex, 16>>;

template <> struct TypeKind<common::TypeCategory::Character, 1> {
  static constexpr common::TypeCategory category{
      common::TypeCategory::Character};
  static constexpr int kind{1};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using CharT = char;
  using StringT = std::basic_string<CharT>;
  using HostT = void;
  using Scalar = value::CharacterValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Character>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

template <> struct TypeKind<common::TypeCategory::Character, 2> {
  static constexpr common::TypeCategory category{
      common::TypeCategory::Character};
  static constexpr int kind{2};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using CharT = char16_t;
  using StringT = std::basic_string<CharT>;
  using HostT = void;
  using Scalar = value::CharacterValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Character>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

template <> struct TypeKind<common::TypeCategory::Character, 4> {
  static constexpr common::TypeCategory category{
      common::TypeCategory::Character};
  static constexpr int kind{4};
  static constexpr int bytesStored{value::IntegerValue::bytesStored(kind)};
  using CharT = char32_t;
  using StringT = std::basic_string<CharT>;
  using HostT = void;
  using Scalar = value::CharacterValue;
  using FortranType = Fortran::evaluate::Type<common::TypeCategory::Character>;
  static constexpr DynamicType GetType() { return DynamicType{category, kind}; }
};

using CharacterKindTypes = std::tuple<TypeKind<TypeCategory::Character, 1>,
    TypeKind<TypeCategory::Character, 2>, TypeKind<TypeCategory::Character, 4>>;

using AllIntrinsicKindTypes =
    common::CombineTuples<IntegerKindTypes, UnsignedKindTypes, LogicalKindTypes,
        RealKindTypes, ComplexKindTypes, CharacterKindTypes>;

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_TYPEKINDTRAITS_H_
