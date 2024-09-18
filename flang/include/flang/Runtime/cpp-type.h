//===-- include/flang/Runtime/cpp-type.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Maps Fortran intrinsic types to C++ types used in the runtime.

#ifndef FORTRAN_RUNTIME_CPP_TYPE_H_
#define FORTRAN_RUNTIME_CPP_TYPE_H_

#include "flang/Common/Fortran.h"
#include "flang/Common/float128.h"
#include "flang/Common/uint128.h"
#include <complex>
#include <cstdint>
#if __cplusplus >= 202302
#include <stdfloat>
#endif
#include <type_traits>

#if !defined HAS_FP16 && __STDCPP_FLOAT16_T__
#define HAS_FP16 1
#endif
#if !defined HAS_BF16 && __STDCPP_BFLOAT16_T__
#define HAS_BF16 1
#endif

namespace Fortran::runtime {

using common::TypeCategory;

template <TypeCategory CAT, int KIND> struct CppTypeForHelper {
  using type = void;
};
template <TypeCategory CAT, int KIND>
using CppTypeFor = typename CppTypeForHelper<CAT, KIND>::type;

template <TypeCategory CAT, int KIND>
constexpr bool HasCppTypeFor{
    !std::is_void_v<typename CppTypeForHelper<CAT, KIND>::type>};

template <int KIND> struct CppTypeForHelper<TypeCategory::Integer, KIND> {
  using type = common::HostSignedIntType<8 * KIND>;
};

#if HAS_FP16
template <> struct CppTypeForHelper<TypeCategory::Real, 2> {
  using type = std::float16_t;
};
#endif
#if HAS_BF16
template <> struct CppTypeForHelper<TypeCategory::Real, 3> {
  using type = std::bfloat16_t;
};
#endif
template <> struct CppTypeForHelper<TypeCategory::Real, 4> {
#if __STDCPP_FLOAT32_T__
  using type = std::float32_t;
#else
  using type = float;
#endif
};
template <> struct CppTypeForHelper<TypeCategory::Real, 8> {
#if __STDCPP_FLOAT64_T__
  using type = std::float64_t;
#else
  using type = double;
#endif
};
#if LDBL_MANT_DIG == 64
template <> struct CppTypeForHelper<TypeCategory::Real, 10> {
  using type = long double;
};
#endif
#if __STDCPP_FLOAT128_T__
using CppFloat128Type = std::float128_t;
#elif LDBL_MANT_DIG == 113
using CppFloat128Type = long double;
#elif HAS_FLOAT128
using CppFloat128Type = __float128;
#endif
#if __STDCPP_FLOAT128_t || LDBL_MANT_DIG == 113 || HAS_FLOAT128
template <> struct CppTypeForHelper<TypeCategory::Real, 16> {
  using type = CppFloat128Type;
};
#endif

template <int KIND> struct CppTypeForHelper<TypeCategory::Complex, KIND> {
  using type = std::complex<CppTypeFor<TypeCategory::Real, KIND>>;
};

template <> struct CppTypeForHelper<TypeCategory::Character, 1> {
  using type = char;
};
template <> struct CppTypeForHelper<TypeCategory::Character, 2> {
  using type = char16_t;
};
template <> struct CppTypeForHelper<TypeCategory::Character, 4> {
  using type = char32_t;
};

template <int KIND> struct CppTypeForHelper<TypeCategory::Logical, KIND> {
  using type = common::HostSignedIntType<8 * KIND>;
};
template <> struct CppTypeForHelper<TypeCategory::Logical, 1> {
  using type = bool;
};

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_CPP_TYPE_H_
