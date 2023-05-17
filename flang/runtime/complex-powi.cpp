/*===-- flang/runtime/complex-powi.cpp ----------------------------*- C++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */
#include "flang/Runtime/entry-names.h"
#include <cstdint>
#include <cstdio>
#include <limits>

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
#else
// on MSVC, C complex is always just a struct of two members as it is not
// supported as a builtin type. So we use C++ complex here as that has the
// same ABI and layout. See:
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/complex-math-support
#include <complex>

// MSVC doesn't allow including <ccomplex> or <complex.h> in C++17 mode to get
// the Windows definitions of these structs so just redefine here.
struct Fcomplex {
  float re;
  float im;
};

struct Dcomplex {
  double re;
  double im;
};

extern "C" Fcomplex RTNAME(cpowi)(Fcomplex base, std::int32_t exp) {
  auto cppbase = *(std::complex<float> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Fcomplex *)(&cppres);
}

extern "C" Dcomplex RTNAME(zpowi)(Dcomplex base, std::int32_t exp) {
  auto cppbase = *(std::complex<double> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Dcomplex *)(&cppres);
}

extern "C" Fcomplex RTNAME(cpowk)(Fcomplex base, std::int64_t exp) {
  auto cppbase = *(std::complex<float> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Fcomplex *)(&cppres);
}

extern "C" Dcomplex RTNAME(zpowk)(Dcomplex base, std::int32_t exp) {
  auto cppbase = *(std::complex<double> *)(&base);
  auto cppres = tgpowi(cppbase, exp);
  return *(Dcomplex *)(&cppres);
}

#endif
