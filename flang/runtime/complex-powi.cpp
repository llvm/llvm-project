/*===-- flang/runtime/complex-powi.cpp ----------------------------*- C++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */
#include "flang/Common/float128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"
#include <cstdint>
#include <cstdio>
#include <limits>

namespace Fortran::runtime {
#ifdef __clang_major__
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif

template <typename C, typename I> C tgpowi(C base, I exp) {
  if (exp == 0) {
    return C{1};
  }

  bool invertResult{exp < 0};
  bool isMin{exp == std::numeric_limits<I>::min()};

  if (isMin) {
    exp = std::numeric_limits<I>::max();
  }

  if (exp < 0) {
    exp = exp * -1;
  }

  C origBase{base};

  while ((exp & 1) == 0) {
    base *= base;
    exp >>= 1;
  }

  C acc{base};

  while (exp > 1) {
    exp >>= 1;
    base *= base;
    if ((exp & 1) == 1) {
      acc *= base;
    }
  }

  if (isMin) {
    acc *= origBase;
  }

  if (invertResult) {
    acc = C{1} / acc;
  }

  return acc;
}

#ifndef _MSC_VER
// With most compilers, C complex is implemented as a builtin type that may have
// specific ABI requirements
extern "C" float _Complex RTNAME(cpowi)(float _Complex base, std::int32_t exp) {
  return tgpowi(base, exp);
}

extern "C" double _Complex RTNAME(zpowi)(
    double _Complex base, std::int32_t exp) {
  return tgpowi(base, exp);
}

extern "C" float _Complex RTNAME(cpowk)(float _Complex base, std::int64_t exp) {
  return tgpowi(base, exp);
}

extern "C" double _Complex RTNAME(zpowk)(
    double _Complex base, std::int64_t exp) {
  return tgpowi(base, exp);
}

#if HAS_LDBL128 || HAS_FLOAT128
// Duplicate CFloat128ComplexType definition from flang/Common/float128.h.
// float128.h does not define it for C++, because _Complex triggers
// c99-extension warnings. We decided to disable warnings for this
// particular file, so we can use _Complex here.
#if HAS_LDBL128
typedef long double _Complex Qcomplex;
#elif HAS_FLOAT128
#if !defined(_ARCH_PPC) || defined(__LONG_DOUBLE_IEEE128__)
typedef _Complex float __attribute__((mode(TC))) Qcomplex;
#else
typedef _Complex float __attribute__((mode(KC))) Qcomplex;
#endif
#endif

extern "C" Qcomplex RTNAME(cqpowi)(Qcomplex base, std::int32_t exp) {
  return tgpowi(base, exp);
}
extern "C" Qcomplex RTNAME(cqpowk)(Qcomplex base, std::int64_t exp) {
  return tgpowi(base, exp);
}
#endif

#else
// on MSVC, C complex is always just a struct of two members as it is not
// supported as a builtin type. So we use C++ complex here as that has the
// same ABI and layout. See:
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/complex-math-support
#include <complex>

// MSVC doesn't allow including <ccomplex> or <complex.h> in C++17 mode to get
// the Windows definitions of these structs so just redefine here.
struct Fcomplex {
  CppTypeFor<TypeCategory::Real, 4> re;
  CppTypeFor<TypeCategory::Real, 4> im;
};

struct Dcomplex {
  CppTypeFor<TypeCategory::Real, 8> re;
  CppTypeFor<TypeCategory::Real, 8> im;
};

extern "C" Fcomplex RTNAME(cpowi)(Fcomplex base, std::int32_t exp) {
  auto cppbase = *(CppTypeFor<TypeCategory::Complex, 4> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Fcomplex *)(&cppres);
}

extern "C" Dcomplex RTNAME(zpowi)(Dcomplex base, std::int32_t exp) {
  auto cppbase = *(CppTypeFor<TypeCategory::Complex, 8> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Dcomplex *)(&cppres);
}

extern "C" Fcomplex RTNAME(cpowk)(Fcomplex base, std::int64_t exp) {
  auto cppbase = *(CppTypeFor<TypeCategory::Complex, 4> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Fcomplex *)(&cppres);
}

extern "C" Dcomplex RTNAME(zpowk)(Dcomplex base, std::int64_t exp) {
  auto cppbase = *(CppTypeFor<TypeCategory::Complex, 8> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Dcomplex *)(&cppres);
}

#if HAS_LDBL128 || HAS_FLOAT128
struct Qcomplex {
  CFloat128Type re;
  CFloat128Type im;
};

extern "C" Dcomplex RTNAME(cqpowi)(Qcomplex base, std::int32_t exp) {
  auto cppbase = *(rtcmplx::complex<CFloat128Type> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Qcomplex *)(&cppres);
}

extern "C" Dcomplex RTNAME(cqpowk)(Qcomplex base, std::int64_t exp) {
  auto cppbase = *(rtcmplx::complex<CFloat128Type> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Qcomplex *)(&cppres);
}
#endif
#endif
} // namespace Fortran::runtime
