//===-- include/flang/Common/erfc-scaled.h-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_ERFC_SCALED_H_
#define FORTRAN_COMMON_ERFC_SCALED_H_

namespace Fortran::common {
template <typename T> inline T ErfcScaled(T arg) {
  // Coefficients for approximation to erfc in the first interval.
  static const T a[5] = {3.16112374387056560e00, 1.13864154151050156e02,
      3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1};
  static const T b[4] = {2.36012909523441209e01, 2.44024637934444173e02,
      1.28261652607737228e03, 2.84423683343917062e03};

  // Coefficients for approximation to erfc in the second interval.
  static const T c[9] = {5.64188496988670089e-1, 8.88314979438837594e00,
      6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02,
      1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03,
      2.15311535474403846e-8};
  static const T d[8] = {1.57449261107098347e01, 1.17693950891312499e02,
      5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03,
      4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03};

  // Coefficients for approximation to erfc in the third interval.
  static const T p[6] = {3.05326634961232344e-1, 3.60344899949804439e-1,
      1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4,
      1.63153871373020978e-2};
  static const T q[5] = {2.56852019228982242e00, 1.87295284992346047e00,
      5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3};

  constexpr T sqrtpi{1.7724538509078120380404576221783883301349L};
  constexpr T rsqrtpi{0.5641895835477562869480794515607725858440L};
  constexpr T epsilonby2{std::numeric_limits<T>::epsilon() * 0.5};
  constexpr T xneg{-26.628e0};
  constexpr T xhuge{6.71e7};
  constexpr T thresh{0.46875e0};
  constexpr T zero{0.0};
  constexpr T one{1.0};
  constexpr T four{4.0};
  constexpr T sixteen{16.0};
  constexpr T xmax{1.0 / (sqrtpi * std::numeric_limits<T>::min())};
  static_assert(xmax > xhuge, "xmax must be greater than xhuge");

  T ysq;
  T xnum;
  T xden;
  T del;
  T result;

  auto x{arg};
  auto y{std::fabs(x)};

  if (y <= thresh) {
    // evaluate erf for  |x| <= 0.46875
    ysq = zero;
    if (y > epsilonby2) {
      ysq = y * y;
    }
    xnum = a[4] * ysq;
    xden = ysq;
    for (int i{0}; i < 3; i++) {
      xnum = (xnum + a[i]) * ysq;
      xden = (xden + b[i]) * ysq;
    }
    result = x * (xnum + a[3]) / (xden + b[3]);
    result = one - result;
    result = std::exp(ysq) * result;
    return result;
  } else if (y <= four) {
    //  evaluate erfc for 0.46875 < |x| <= 4.0
    xnum = c[8] * y;
    xden = y;
    for (int i{0}; i < 7; ++i) {
      xnum = (xnum + c[i]) * y;
      xden = (xden + d[i]) * y;
    }
    result = (xnum + c[7]) / (xden + d[7]);
  } else {
    //  evaluate erfc for |x| > 4.0
    result = zero;
    if (y >= xhuge) {
      if (y < xmax) {
        result = rsqrtpi / y;
      }
    } else {
      ysq = one / (y * y);
      xnum = p[5] * ysq;
      xden = ysq;
      for (int i{0}; i < 4; ++i) {
        xnum = (xnum + p[i]) * ysq;
        xden = (xden + q[i]) * ysq;
      }
      result = ysq * (xnum + p[4]) / (xden + q[4]);
      result = (rsqrtpi - result) / y;
    }
  }
  //  fix up for negative argument, erf, etc.
  if (x < zero) {
    if (x < xneg) {
      result = std::numeric_limits<T>::max();
    } else {
      ysq = trunc(x * sixteen) / sixteen;
      del = (x - ysq) * (x + ysq);
      y = std::exp((ysq * ysq)) * std::exp((del));
      result = (y + y) - result;
    }
  }
  return result;
}
} // namespace Fortran::common
#endif // FORTRAN_COMMON_ERFC_SCALED_H_
