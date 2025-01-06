//===-- complex type --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_COMPLEX_TYPE_H
#define LLVM_LIBC_SRC___SUPPORT_COMPLEX_TYPE_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/complex_types.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
template <typename T> struct Complex {
  T real;
  T imag;
};

template <typename T> struct make_complex;

template <> struct make_complex<float> {
  using type = _Complex float;
};
template <> struct make_complex<double> {
  using type = _Complex double;
};
template <> struct make_complex<long double> {
  using type = _Complex long double;
};

#if defined(LIBC_TYPES_HAS_CFLOAT16)
template <> struct make_complex<float16> {
  using type = cfloat16;
};
#endif
#ifdef LIBC_TYPES_CFLOAT128_IS_NOT_COMPLEX_LONG_DOUBLE
template <> struct make_complex<float128> {
  using type = cfloat128;
};
#endif

template <typename T> using make_complex_t = typename make_complex<T>::type;

template <typename T> struct make_real;

template <> struct make_real<_Complex float> {
  using type = float;
};
template <> struct make_real<_Complex double> {
  using type = double;
};
template <> struct make_real<_Complex long double> {
  using type = long double;
};

#if defined(LIBC_TYPES_HAS_CFLOAT16)
template <> struct make_real<cfloat16> {
  using type = float16;
};
#endif
#ifdef LIBC_TYPES_CFLOAT128_IS_NOT_COMPLEX_LONG_DOUBLE
template <> struct make_real<cfloat128> {
  using type = float128;
};
#endif

template <typename T> using make_real_t = typename make_real<T>::type;

template <typename T> LIBC_INLINE constexpr T conjugate(T c) {
  Complex<make_real_t<T>> c_c = cpp::bit_cast<Complex<make_real_t<T>>>(c);
  c_c.imag = -c_c.imag;
  return cpp::bit_cast<T>(c_c);
}

template <typename T> LIBC_INLINE constexpr T project(T c) {
  using real_t = make_real_t<T>;
  Complex<real_t> c_c = cpp::bit_cast<Complex<real_t>>(c);
  if (fputil::FPBits<real_t>(c_c.real).is_inf() ||
      fputil::FPBits<real_t>(c_c.imag).is_inf()) {
    return cpp::bit_cast<T>(
        Complex<real_t>{(fputil::FPBits<real_t>::inf(Sign::POS).get_val()),
                        static_cast<real_t>(c_c.imag > 0 ? 0.0 : -0.0)});
  } else {
    return c;
  }
}

} // namespace LIBC_NAMESPACE_DECL
#endif // LLVM_LIBC_SRC___SUPPORT_COMPLEX_TYPE_H
