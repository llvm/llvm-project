//===-- include/flang/Evaluate/pgmath.h.inc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines the interface of libpgmath to be used for folding
// and code generation.
// Usage:
// define PGMATH_DECLARE if you simply want to declare pgmath interface.
// Define the PGMATH_USE_S/D/C/Z(intrinsic name, function name) according
// to what needs to be done with the runtime declaration.
// This intrinsic will be called on all libpgmath function in the
// intrinsic alphabetical order.
// Define PGMATH_FAST/RELAXED/PRECISE to restrict the PGMATH_USE visit
// to the targeted versions.
// Define PGMATH_USE_OTHER to visit math functions that are not related to
// floating points (e.g. int**int pow).

// Control Macros
#ifdef PGMATH_DECLARE
#undef PGMATH_DECLARE
#define DEFINE_C_COMPLEX_TYPES
#define PGMATH_DECLARE(x) extern "C" x;
#define PGMATH_FAST
#define PGMATH_PRECISE
#define PGMATH_RELAXED
#else
#define PGMATH_DECLARE(x)
#endif

#ifdef PGMATH_USE_ALL_TYPES
#define PGMATH_USE_S(name, func) PGMATH_USE_ALL_TYPES(name, func)
#define PGMATH_USE_D(name, func) PGMATH_USE_ALL_TYPES(name, func)
#define PGMATH_USE_C(name, func) PGMATH_USE_ALL_TYPES(name, func)
#define PGMATH_USE_Z(name, func) PGMATH_USE_ALL_TYPES(name, func)
#define PGMATH_USE_OTHER(name, func) PGMATH_USE_ALL_TYPES(name, func)
#endif

#ifndef PGMATH_USE_S
#define PGMATH_USE_S(name, x)
#endif

#ifndef PGMATH_USE_D
#define PGMATH_USE_D(name, x)
#endif

#ifndef PGMATH_USE_C
#define PGMATH_USE_C(name, x)
#endif

#ifndef PGMATH_USE_Z
#define PGMATH_USE_Z(name, x)
#endif

#ifndef PGMATH_USE_OTHER
#define PGMATH_USE_OTHER(name, x)
#endif

// Handle the C99 _Complex vs C++ std::complex call interface issue.
// _Complex and std::complex are layout compatible (they are the same when
// in memory), but they are not guaranteed to be compatible in call interface
// (they may be passed/returned differently). For instance on X86 32 bits,
// float _complex is returned in a pair of register, but std::complex<float>
// is returned in memory.
// Pgmath is defined in C using _Complex (and windows _Fcomplex/_DComplex
// equivalents). Since this file defines the call interface with the runtime
// for both folding and code generation (through template introspection), it
// is crucial to make a distinction between std::complex and _Complex here.
// Unfortunately, _Complex support is not standard in C++.
// Reserve pgmath usage at compile time (folding) when _Complex is available
// (cmake is responsible to detect this).
// For code generation, define type c_float_complex_t that can be used in
// introspection to indicate that the C99 _Complex ABI has to be used for the
// related value.
#ifdef DEFINE_C_COMPLEX_TYPES
#ifdef PGMATH_LINKING
#ifdef _WIN32
using c_float_complex_t = _Fcomplex;
using c_double_complex_t = _Dcomplex;
#else
using c_float_complex_t = float _Complex;
using c_double_complex_t = double _Complex;
#endif
#else
struct c_float_complex_t {};
struct c_double_complex_t {};
#endif
#endif

#define PGMATH_REAL_IMPL(impl, func) \
  PGMATH_DECLARE(float __##impl##s_##func##_1(float)) \
  PGMATH_DECLARE(double __##impl##d_##func##_1(double)) \
  PGMATH_USE_S(func, __##impl##s_##func##_1) \
  PGMATH_USE_D(func, __##impl##d_##func##_1)

#define PGMATH_COMPLEX_IMPL(impl, func) \
  PGMATH_DECLARE(c_float_complex_t __##impl##c_##func##_1(c_float_complex_t)) \
  PGMATH_DECLARE( \
      c_double_complex_t __##impl##z_##func##_1(c_double_complex_t)) \
  PGMATH_USE_C(func, __##impl##c_##func##_1) \
  PGMATH_USE_Z(func, __##impl##z_##func##_1)

#define PGMATH_ALL_FP_IMPL(impl, func) \
  PGMATH_REAL_IMPL(impl, func) \
  PGMATH_FAST_COMPLEX_IMPL(impl, func)

#define PGMATH_REAL2_IMPL(impl, func) \
  PGMATH_DECLARE(float __##impl##s_##func##_1(float, float)) \
  PGMATH_DECLARE(double __##impl##d_##func##_1(double, double)) \
  PGMATH_USE_S(func, __##impl##s_##func##_1) \
  PGMATH_USE_D(func, __##impl##d_##func##_1)

#define PGMATH_COMPLEX2_IMPL(impl, func) \
  PGMATH_DECLARE(c_float_complex_t __##impl##c_##func##_1( \
      c_float_complex_t, c_float_complex_t)) \
  PGMATH_DECLARE(c_double_complex_t __##impl##z_##func##_1( \
      c_double_complex_t, c_double_complex_t)) \
  PGMATH_USE_C(func, __##impl##c_##func##_1) \
  PGMATH_USE_Z(func, __##impl##z_##func##_1)

#define PGMATH_ALL_FP2_IMPL(impl, func) \
  PGMATH_REAL2_IMPL(func) \
  PGMATH_COMPLEX2_IMPL(func)

#undef PGMATH_FAST_REAL
#undef PGMATH_FAST_COMPLEX
#undef PGMATH_FAST_ALL_FP
#undef PGMATH_FAST_REAL2
#undef PGMATH_FAST_COMPLEX2
#undef PGMATH_FAST_ALL_FP2
#ifdef PGMATH_FAST
#define PGMATH_FAST_REAL(func) PGMATH_REAL_IMPL(f, func)
#define PGMATH_FAST_COMPLEX(func) PGMATH_COMPLEX_IMPL(f, func)
#define PGMATH_FAST_ALL_FP(func) PGMATH_ALL_IMPL(f, func)
#define PGMATH_FAST_REAL2(func) PGMATH_REAL2_IMPL(f, func)
#define PGMATH_FAST_COMPLEX2(func) PGMATH_COMPLEX2_IMPL(f, func)
#define PGMATH_FAST_ALL_FP2(func) PGMATH_ALL_FP2_IMPL(f, func)
#else
#define PGMATH_FAST_REAL(func)
#define PGMATH_FAST_COMPLEX(func)
#define PGMATH_FAST_ALL_FP(func)
#define PGMATH_FAST_REAL2(func)
#define PGMATH_FAST_COMPLEX2(func)
#define PGMATH_FAST_ALL_FP2(func)
#endif

#undef PGMATH_RELAXED_REAL
#undef PGMATH_RELAXED_COMPLEX
#undef PGMATH_RELAXED_ALL_FP
#undef PGMATH_RELAXED_REAL2
#undef PGMATH_RELAXED_COMPLEX2
#undef PGMATH_RELAXED_ALL_FP2
#ifdef PGMATH_RELAXED
#define PGMATH_RELAXED_REAL(func) PGMATH_REAL_IMPL(r, func)
#define PGMATH_RELAXED_COMPLEX(func) PGMATH_COMPLEX_IMPL(r, func)
#define PGMATH_RELAXED_ALL_FP(func) PGMATH_ALL_IMPL(r, func)
#define PGMATH_RELAXED_REAL2(func) PGMATH_REAL2_IMPL(r, func)
#define PGMATH_RELAXED_COMPLEX2(func) PGMATH_COMPLEX2_IMPL(r, func)
#define PGMATH_RELAXED_ALL_FP2(func) PGMATH_ALL_FP2_IMPL(r, func)
#else
#define PGMATH_RELAXED_REAL(func)
#define PGMATH_RELAXED_COMPLEX(func)
#define PGMATH_RELAXED_ALL_FP(func)
#define PGMATH_RELAXED_REAL2(func)
#define PGMATH_RELAXED_COMPLEX2(func)
#define PGMATH_RELAXED_ALL_FP2(func)
#endif

#undef PGMATH_PRECISE_REAL
#undef PGMATH_PRECISE_COMPLEX
#undef PGMATH_PRECISE_ALL_FP
#undef PGMATH_PRECISE_REAL2
#undef PGMATH_PRECISE_COMPLEX2
#undef PGMATH_PRECISE_ALL_FP2
#ifdef PGMATH_PRECISE
#define PGMATH_PRECISE_REAL(func) PGMATH_REAL_IMPL(p, func)
#define PGMATH_PRECISE_COMPLEX(func) PGMATH_COMPLEX_IMPL(p, func)
#define PGMATH_PRECISE_ALL_FP(func) PGMATH_ALL_IMPL(p, func)
#define PGMATH_PRECISE_REAL2(func) PGMATH_REAL2_IMPL(p, func)
#define PGMATH_PRECISE_COMPLEX2(func) PGMATH_COMPLEX2_IMPL(p, func)
#define PGMATH_PRECISE_ALL_FP2(func) PGMATH_ALL_FP2_IMPL(p, func)
#else
#define PGMATH_PRECISE_REAL(func)
#define PGMATH_PRECISE_COMPLEX(func)
#define PGMATH_PRECISE_ALL_FP(func)
#define PGMATH_PRECISE_REAL2(func)
#define PGMATH_PRECISE_COMPLEX2(func)
#define PGMATH_PRECISE_ALL_FP2(func)
#endif

#define PGMATH_REAL(func) \
  PGMATH_FAST_REAL(func) \
  PGMATH_PRECISE_REAL(func) \
  PGMATH_RELAXED_REAL(func)

#define PGMATH_COMPLEX(func) \
  PGMATH_FAST_COMPLEX(func) \
  PGMATH_PRECISE_COMPLEX(func) \
  PGMATH_RELAXED_COMPLEX(func)

#define PGMATH_ALL(func) \
  PGMATH_REAL(func) \
  PGMATH_COMPLEX(func)

#define PGMATH_REAL2(func) \
  PGMATH_FAST_REAL2(func) \
  PGMATH_PRECISE_REAL2(func) \
  PGMATH_RELAXED_REAL2(func)

#define PGMATH_COMPLEX2(func) \
  PGMATH_FAST_COMPLEX2(func) \
  PGMATH_PRECISE_COMPLEX2(func) \
  PGMATH_RELAXED_COMPLEX2(func)

#define PGMATH_ALL2(func) \
  PGMATH_REAL2(func) \
  PGMATH_COMPLEX2(func)

// Marcos to declare __mth_i libpgmath variants
#define PGMATH_MTH_VERSION_REAL(func) \
  PGMATH_DECLARE(float __mth_i_##func(float)) \
  PGMATH_DECLARE(double __mth_i_d##func(double)) \
  PGMATH_USE_S(func, __mth_i_##func) \
  PGMATH_USE_D(func, __mth_i_d##func)

// Actual libpgmath declarations
PGMATH_ALL(acos)
PGMATH_MTH_VERSION_REAL(acosh)
PGMATH_ALL(asin)
PGMATH_MTH_VERSION_REAL(asinh)
PGMATH_ALL(atan)
PGMATH_REAL2(atan2)
PGMATH_MTH_VERSION_REAL(atanh)
PGMATH_MTH_VERSION_REAL(bessel_j0)
PGMATH_MTH_VERSION_REAL(bessel_j1)
// bessel_jn and bessel_yn takes an int as first arg
PGMATH_DECLARE(float __mth_i_bessel_jn(int, float))
PGMATH_DECLARE(double __mth_i_dbessel_jn(int, double))
PGMATH_USE_S(bessel_jn, __mth_i_bessel_jn)
PGMATH_USE_D(bessel_jn, __mth_i_dbessel_jn)
PGMATH_MTH_VERSION_REAL(bessel_y0)
PGMATH_MTH_VERSION_REAL(bessel_y1)
PGMATH_DECLARE(float __mth_i_bessel_yn(int, float))
PGMATH_DECLARE(double __mth_i_dbessel_yn(int, double))
PGMATH_USE_S(bessel_yn, __mth_i_bessel_yn)
PGMATH_USE_D(bessel_yn, __mth_i_dbessel_yn)

PGMATH_ALL(cos)
PGMATH_ALL(cosh)
PGMATH_MTH_VERSION_REAL(erf)
PGMATH_MTH_VERSION_REAL(erfc)
PGMATH_MTH_VERSION_REAL(erfc_scaled)
PGMATH_ALL(exp)
PGMATH_MTH_VERSION_REAL(gamma)

PGMATH_DECLARE(float __mth_i_hypot(float, float))
PGMATH_DECLARE(double __mth_i_dhypot(double, double))
PGMATH_USE_S(hypot, __mth_i_hypot)
PGMATH_USE_D(hypot, __mth_i_dhypot)

PGMATH_ALL(log)
PGMATH_REAL(log10)
PGMATH_MTH_VERSION_REAL(log_gamma)
// no function for modulo in libpgmath.
// fast mod used in all versions.
PGMATH_DECLARE(float __fs_mod_1(float, float))
PGMATH_DECLARE(double __fd_mod_1(double, double))
PGMATH_USE_S(mod, __fs_mod_1)
PGMATH_USE_D(mod, __fd_mod_1)

PGMATH_ALL2(pow)
// Versions of pow with integer exponents
#define PGMATH_DELCARE_POW(impl) \
  PGMATH_DECLARE(float __##impl##s_powi_1(float, int)) \
  PGMATH_DECLARE(double __##impl##d_powi_1(double, int)) \
  PGMATH_DECLARE(c_float_complex_t __##impl##c_powi_1(c_float_complex_t, int)) \
  PGMATH_DECLARE( \
      c_double_complex_t __##impl##z_powi_1(c_double_complex_t, int)) \
  PGMATH_USE_S(pow, __##impl##s_powi_1) \
  PGMATH_USE_D(pow, __##impl##d_powi_1) \
  PGMATH_USE_C(pow, __##impl##c_powi_1) \
  PGMATH_USE_Z(pow, __##impl##z_powi_1) \
  PGMATH_DECLARE(float __##impl##s_powk_1(float, int64_t)) \
  PGMATH_DECLARE(double __##impl##d_powk_1(double, int64_t)) \
  PGMATH_DECLARE( \
      c_float_complex_t __##impl##c_powk_1(c_float_complex_t, int64_t)) \
  PGMATH_DECLARE( \
      c_double_complex_t __##impl##z_powk_1(c_double_complex_t, int64_t)) \
  PGMATH_USE_S(pow, __##impl##s_powk_1) \
  PGMATH_USE_D(pow, __##impl##d_powk_1) \
  PGMATH_USE_C(pow, __##impl##c_powk_1) \
  PGMATH_USE_Z(pow, __##impl##z_powk_1)

#ifdef PGMATH_FAST
PGMATH_DELCARE_POW(f)
#endif
#ifdef PGMATH_RELAXED
PGMATH_DELCARE_POW(r)
#endif
#ifdef PGMATH_PRECISE
PGMATH_DELCARE_POW(p)
#endif

// integer ** integer versions of pow
PGMATH_DECLARE(int __mth_i_ipowi(int, int))
PGMATH_DECLARE(int64_t __mth_i_kpowk(int64_t, int64_t))
PGMATH_USE_OTHER(pow, __mth_i_ipowi)
PGMATH_USE_OTHER(pow, __mth_i_kpowk)

PGMATH_ALL(sin)
PGMATH_ALL(sinh)
PGMATH_MTH_VERSION_REAL(sqrt)
PGMATH_COMPLEX(sqrt) // real versions are __mth_i...
PGMATH_ALL(tan)
PGMATH_ALL(tanh)

#undef PGMATH_DECLARE
#undef PGMATH_FAST
#undef PGMATH_PRECISE
#undef PGMATH_RELAXED
#undef PGMATH_USE_S
#undef PGMATH_USE_D
#undef PGMATH_USE_C
#undef PGMATH_USE_Z
#undef PGMATH_USE_OTHER
#undef PGMATH_USE_ALL_TYPES
#undef PGMATH_LINKING
#undef DEFINE_C_COMPLEX_TYPES
