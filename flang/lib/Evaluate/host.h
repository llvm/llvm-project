//===-- lib/Evaluate/host.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_HOST_H_
#define FORTRAN_EVALUATE_HOST_H_

// Define a compile-time mapping between Fortran intrinsic types and host
// hardware types if possible. The purpose is to avoid having to do any kind of
// assumption on whether a "float" matches the Scalar<Type<TypeCategory::Real,
// 4>> outside of this header. The main tools are HostTypeExists<T> and
// HostType<T>. HostTypeExists<T>() will return true if and only if a host
// hardware type maps to Fortran intrinsic type T. Then HostType<T> can be used
// to safely refer to this hardware type.

#if HAS_QUADMATHLIB
#include "quadmath_wrapper.h"
#include "flang/Common/float128.h"
#endif
#include "flang/Evaluate/type.h"
#include <cfenv>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace Fortran::evaluate {
namespace host {

// Helper class to handle host runtime traps, status flag and errno
class HostFloatingPointEnvironment {
public:
  void SetUpHostFloatingPointEnvironment(FoldingContext &);
  void CheckAndRestoreFloatingPointEnvironment(FoldingContext &);
  bool hasSubnormalFlushingHardwareControl() const {
    return hasSubnormalFlushingHardwareControl_;
  }
  void SetFlag(RealFlag flag) { flags_.set(flag); }
  bool hardwareFlagsAreReliable() const { return hardwareFlagsAreReliable_; }

private:
  std::fenv_t originalFenv_;
#if __x86_64__
  unsigned int originalMxcsr;
#endif
  RealFlags flags_;
  bool hasSubnormalFlushingHardwareControl_{false};
  bool hardwareFlagsAreReliable_{true};
};

// Type mapping from F18 types to host types
struct UnsupportedType {}; // There is no host type for the F18 type

// Compile-time (category, kind) tag used only within this header.  Since the
// migration to runtime kinds, Fortran's Type<CAT> no longer carries its kind
// as a template parameter (kind is a runtime property of the value).  The
// host-folding layer, however, still needs a distinct compile-time type per
// (category, kind) so that it can be mapped to a concrete host C++ type (and
// back) and so that the per-kind host runtime function tables can be
// enumerated.  TypeKind also exposes the kindless Fortran Type<CAT> and its
// runtime-kind Scalar so that generic helpers (GetScalarConstantValue,
// Constant, ...) can be reached through ::FortranType.
template <common::TypeCategory CAT, int KIND> struct TypeKind {
  static constexpr common::TypeCategory category{CAT};
  static constexpr int kind{KIND};
  using FortranType = Fortran::evaluate::Type<CAT>;
  using Scalar = Fortran::evaluate::Scalar<FortranType>;
  // Meaningful for COMPLEX only: the real component's (category, kind) tag.
  using Part = TypeKind<common::TypeCategory::Real, KIND>;
  static constexpr DynamicType GetType() { return DynamicType{CAT, KIND}; }
};

template <typename FTN_T> struct HostTypeHelper {
  using Type = UnsupportedType;
};
template <typename FTN_T> using HostType = typename HostTypeHelper<FTN_T>::Type;

template <typename... T> constexpr inline bool HostTypeExists() {
  return (... && (!std::is_same_v<HostType<T>, UnsupportedType>));
}

template <typename, typename = void> struct HasValueMethod : std::false_type {};
template <typename T>
struct HasValueMethod<T,
    std::void_t<decltype(std::declval<const T &>().value())>> : std::true_type {
};

template <typename, typename = void> struct HasRawBits : std::false_type {};
template <typename T>
struct HasRawBits<T, std::void_t<decltype(std::declval<const T &>().RawBits())>>
    : std::true_type {};

// Type mapping from host types to F18 types FortranType<HOST_T> is defined
// after all HosTypeHelper definition because it reverses them to avoid
// duplication.

// Scalar conversion utilities from host scalars to F18 scalars
template <typename FTN_T>
inline constexpr Scalar<FTN_T> CastHostToFortran(const HostType<FTN_T> &x) {
  static_assert(HostTypeExists<FTN_T>());
  if constexpr (FTN_T::category == TypeCategory::Complex &&
      sizeof(Scalar<FTN_T>) != sizeof(HostType<FTN_T>)) {
    // X87 is usually padded to 12 or 16bytes. Need to cast piecewise for
    // complex
#if HAS_QUADMATHLIB
    if constexpr (std::is_same_v<HostType<FTN_T>, __complex128>) {
      return Scalar<FTN_T>{CastHostToFortran<typename FTN_T::Part>(__real__ x),
          CastHostToFortran<typename FTN_T::Part>(__imag__ x)};
    } else
#endif
      return Scalar<FTN_T>{
          CastHostToFortran<typename FTN_T::Part>(std::real(x)),
          CastHostToFortran<typename FTN_T::Part>(std::imag(x))};
  } else if constexpr (FTN_T::category == TypeCategory::Real &&
      HasRawBits<Scalar<FTN_T>>::value) {
    return Scalar<FTN_T>::FromRawBytes(
        &x, std::min(sizeof x, static_cast<std::size_t>(FTN_T::kind)));
  } else if constexpr (FTN_T::category == TypeCategory::Real &&
      HasValueMethod<Scalar<FTN_T>>::value) {
    using ValueWord = typename Scalar<FTN_T>::Word;
    return Scalar<FTN_T>{
        ValueWord::FromRawBytes(
            &x, std::min(sizeof x, static_cast<std::size_t>(FTN_T::kind))),
        FTN_T::kind};
  } else if constexpr ((FTN_T::category == TypeCategory::Integer ||
                           FTN_T::category == TypeCategory::Unsigned) &&
      value::HasWord<Scalar<FTN_T>>::value) {
    return Scalar<FTN_T>{x};
  } else {
    return *reinterpret_cast<const Scalar<FTN_T> *>(&x);
  }
}

// Scalar conversion utilities from F18 scalars to host scalars.
template <typename FTN_T>
inline constexpr HostType<FTN_T> CastFortranToHost(const Scalar<FTN_T> &x) {
  static_assert(HostTypeExists<FTN_T>());
  if constexpr (FTN_T::category == TypeCategory::Complex) {
    using FortranPartType = typename FTN_T::Part;
#if HAS_QUADMATHLIB
    if constexpr (std::is_same_v<HostType<FTN_T>, __complex128>) {
      HostType<FTN_T> y;
      __real__ y = CastFortranToHost<FortranPartType>(x.REAL());
      __imag__ y = CastFortranToHost<FortranPartType>(x.AIMAG());
      return y;
    } else
#endif
      return HostType<FTN_T>{CastFortranToHost<FortranPartType>(x.REAL()),
          CastFortranToHost<FortranPartType>(x.AIMAG())};
  } else if constexpr (std::is_same_v<FTN_T,
                           TypeKind<TypeCategory::Real, 10>> &&
      !HasValueMethod<Scalar<FTN_T>>::value &&
      !HasRawBits<Scalar<FTN_T>>::value) {
    // x87 80-bit floating-point occupies 16 bytes as a C "long double";
    // copy the data to avoid a legitimate (but benign due to little-endianness)
    // warning from GCC >= 11.2.0.
    HostType<FTN_T> y;
    std::memcpy(&y, &x, sizeof x);
    return y;
  } else if constexpr (FTN_T::category == TypeCategory::Real &&
      HasRawBits<Scalar<FTN_T>>::value) {
    HostType<FTN_T> y{};
    x.StoreRawBytes(&y, sizeof y);
    return y;
  } else if constexpr (FTN_T::category == TypeCategory::Real &&
      HasValueMethod<Scalar<FTN_T>>::value) {
    HostType<FTN_T> y{};
    x.RawBits().StoreRawBytes(&y, FTN_T::kind);
    return y;
  } else if constexpr ((FTN_T::category == TypeCategory::Integer ||
                           FTN_T::category == TypeCategory::Unsigned) &&
      value::HasWord<Scalar<FTN_T>>::value) {
    if constexpr (std::is_signed_v<HostType<FTN_T>>) {
      return x.template ToSInt<HostType<FTN_T>>();
    } else {
      return static_cast<HostType<FTN_T>>(x.ToUInt64());
    }
  } else {
    static_assert(sizeof x == sizeof(HostType<FTN_T>));
    return *reinterpret_cast<const HostType<FTN_T> *>(&x);
  }
}

template <> struct HostTypeHelper<TypeKind<TypeCategory::Integer, 1>> {
  using Type = std::int8_t;
};

template <> struct HostTypeHelper<TypeKind<TypeCategory::Integer, 2>> {
  using Type = std::int16_t;
};

template <> struct HostTypeHelper<TypeKind<TypeCategory::Integer, 4>> {
  using Type = std::int32_t;
};

template <> struct HostTypeHelper<TypeKind<TypeCategory::Integer, 8>> {
  using Type = std::int64_t;
};

template <> struct HostTypeHelper<TypeKind<TypeCategory::Integer, 16>> {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__SIZEOF_INT128__)
  using Type = __int128_t;
#else
  using Type = UnsupportedType;
#endif
};

// TODO no mapping to host types are defined currently for 16bits float
// It should be defined when gcc/clang have a better support for it.

template <>
struct HostTypeHelper<
    TypeKind<TypeCategory::Real, common::RealKindForPrecision(24)>> {
  // IEEE 754 32bits
  using Type = std::conditional_t<sizeof(float) == 4 &&
          std::numeric_limits<float>::is_iec559,
      float, UnsupportedType>;
};

template <>
struct HostTypeHelper<
    TypeKind<TypeCategory::Real, common::RealKindForPrecision(53)>> {
  // IEEE 754 64bits
  using Type = std::conditional_t<sizeof(double) == 8 &&
          std::numeric_limits<double>::is_iec559,
      double, UnsupportedType>;
};

template <>
struct HostTypeHelper<
    TypeKind<TypeCategory::Real, common::RealKindForPrecision(64)>> {
  // X87 80bits
  using Type = std::conditional_t<sizeof(long double) >= 10 &&
          std::numeric_limits<long double>::digits == 64 &&
          std::numeric_limits<long double>::max_exponent == 16384,
      long double, UnsupportedType>;
};

#if HAS_QUADMATHLIB
template <> struct HostTypeHelper<TypeKind<TypeCategory::Real, 16>> {
  // IEEE 754 128bits
  using Type = __float128;
};
#else
template <> struct HostTypeHelper<TypeKind<TypeCategory::Real, 16>> {
  // IEEE 754 128bits
  using Type = std::conditional_t<sizeof(long double) == 16 &&
          std::numeric_limits<long double>::digits == 113 &&
          std::numeric_limits<long double>::max_exponent == 16384,
      long double, UnsupportedType>;
};
#endif

template <int KIND>
struct HostTypeHelper<TypeKind<TypeCategory::Complex, KIND>> {
  using RealT = TypeKind<TypeCategory::Real, KIND>;
  using Type = std::conditional_t<HostTypeExists<RealT>(),
      std::complex<HostType<RealT>>, UnsupportedType>;
};

#if HAS_QUADMATHLIB
template <> struct HostTypeHelper<TypeKind<TypeCategory::Complex, 16>> {
  using RealT = TypeKind<TypeCategory::Real, 16>;
  using Type = __complex128;
};
#endif

template <int KIND>
struct HostTypeHelper<TypeKind<TypeCategory::Logical, KIND>> {
  using Type = std::conditional_t<KIND <= 8, std::uint8_t, UnsupportedType>;
};

template <int KIND>
struct HostTypeHelper<TypeKind<TypeCategory::Character, KIND>> {
  using Type = typename TypeKind<TypeCategory::Character, KIND>::Scalar;
};

// Enumerates every (category, kind) tag the host-folding layer understands.
// This replaces the former reliance on AllIntrinsicTypes, which no longer
// distinguishes kinds at the type level.
using AllHostKindTypes = std::tuple<TypeKind<TypeCategory::Integer, 1>,
    TypeKind<TypeCategory::Integer, 2>, TypeKind<TypeCategory::Integer, 4>,
    TypeKind<TypeCategory::Integer, 8>, TypeKind<TypeCategory::Integer, 16>,
    TypeKind<TypeCategory::Real, 2>, TypeKind<TypeCategory::Real, 3>,
    TypeKind<TypeCategory::Real, 4>, TypeKind<TypeCategory::Real, 8>,
    TypeKind<TypeCategory::Real, 10>, TypeKind<TypeCategory::Real, 16>,
    TypeKind<TypeCategory::Complex, 2>, TypeKind<TypeCategory::Complex, 3>,
    TypeKind<TypeCategory::Complex, 4>, TypeKind<TypeCategory::Complex, 8>,
    TypeKind<TypeCategory::Complex, 10>, TypeKind<TypeCategory::Complex, 16>,
    TypeKind<TypeCategory::Logical, 1>, TypeKind<TypeCategory::Logical, 2>,
    TypeKind<TypeCategory::Logical, 4>, TypeKind<TypeCategory::Logical, 8>,
    TypeKind<TypeCategory::Character, 1>, TypeKind<TypeCategory::Character, 2>,
    TypeKind<TypeCategory::Character, 4>>;

// Type mapping from host types to F18 types. This need to be placed after all
// HostTypeHelper specializations.
template <typename T, typename... TT> struct IndexInTupleHelper {};
template <typename T, typename... TT>
struct IndexInTupleHelper<T, std::tuple<TT...>> {
  static constexpr int value{common::TypeIndex<T, TT...>};
};
struct UnknownType {}; // the host type does not match any F18 types
template <typename HOST_T> struct FortranTypeHelper {
  using HostTypeMapping =
      common::MapTemplate<HostType, AllHostKindTypes, std::tuple>;
  static constexpr int index{
      IndexInTupleHelper<HOST_T, HostTypeMapping>::value};
  // Both conditional types are "instantiated", so a valid type must be
  // created for invalid index even if not used.
  using Type = std::conditional_t<index >= 0,
      std::tuple_element_t<(index >= 0) ? index : 0, AllHostKindTypes>,
      UnknownType>;
};

template <typename HOST_T>
using FortranType = typename FortranTypeHelper<HOST_T>::Type;

template <typename... HT> constexpr inline bool FortranTypeExists() {
  return (... && (!std::is_same_v<FortranType<HT>, UnknownType>));
}

} // namespace host
} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_HOST_H_
