//===-- Implementation header for lgammaf -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_LGAMMAF_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_LGAMMAF_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/gamma_util.h"
#include "src/__support/math/log.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE float lgammaf(float x) {
  using namespace gamma_internal;
  using FPBits = fputil::FPBits<float>;

  FPBits xbits(x);
  uint32_t x_abs = xbits.abs().uintval();

  // NaN / Inf
  if (LIBC_UNLIKELY(x_abs >= 0x7f800000u)) {
    if (x_abs == 0x7f800000u)
      return FPBits::inf().get_val();
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // +/- 0 -> +Inf pole
  if (LIBC_UNLIKELY(x_abs == 0)) {
    fputil::raise_except_if_required(FE_DIVBYZERO);
    fputil::set_errno_if_required(ERANGE);
    return FPBits::inf().get_val();
  }

  // Negative integers and lgamma(1) = lgamma(2) = 0.
  if (LIBC_UNLIKELY(is_integer(x))) {
    if (xbits.is_neg()) {
      fputil::raise_except_if_required(FE_DIVBYZERO);
      fputil::set_errno_if_required(ERANGE);
      return FPBits::inf().get_val();
    }
    if (x_abs == 0x3f800000u || x_abs == 0x40000000u)
      return FPBits::zero().get_val();
  }

  constexpr fputil::ExceptValues<float, 11> LGAMMAF_EXCEPTS{{
      // input,      toward-zero result, RU, RD, RN
      {0x3b7c53aau, 0x40b1d661u, 1, 0, 0},
      {0x42468b59u, 0x430f25a7u, 1, 0, 0},
      {0x50522f52u, 0x5292ee5du, 1, 0, 1},
      {0x9b7679ffu, 0x4247c72cu, 1, 0, 1},
      {0x9e88452du, 0x4236bd8bu, 1, 0, 1},
      {0xa77a8e47u, 0x42052b94u, 1, 0, 1},
      {0xb0e17820u, 0x41a1d37bu, 1, 0, 0},
      {0xc02c060fu, 0xbdabc3e2u, 0, 1, 1},
      {0xc134eb14u, 0xc1875615u, 0, 1, 0},
      {0xc33139a3u, 0xc43991afu, 0, 1, 0},
      {0xc6f7e151u, 0xc89116deu, 0, 1, 0},
  }};
  if (auto r = LGAMMAF_EXCEPTS.lookup(xbits.uintval());
      LIBC_UNLIKELY(r.has_value()))
    return r.value();

  double xd = fputil::cast<double>(x);
  double abs_xd = xd < 0.0 ? -xd : xd;
  double lgamma_val;

  // For very tiny |x| (< 2^-23), use the truncated Laurent series:
  //   lgamma(x)  = -log(x) - gamma*x + O(x^2)  for tiny x > 0
  //   lgamma(-y) = -log(y) + gamma*y + O(y^2)  for tiny y > 0
  // Even though gamma*|x| < 2^-25 is below 1 float ULP of |result|, it tips
  // rounding at boundary cases. The x^2 term is at most gamma*2^-46 << 2^-50.
  if (x_abs < 0x34000000u) { // |x| < 2^-23
    constexpr double EULER_GAMMA = 0x1.2788cfc6fb619p-1;
    double sign_corr = xbits.is_neg() ? EULER_GAMMA : -EULER_GAMMA;
    return fputil::cast<float>(
        fputil::multiply_add(sign_corr, abs_xd, -math::log(abs_xd)));
  }

  if (x_abs < 0x3f290000u) {
    // Small: t = |x| < 0.66015625. Degree-22 Chebyshev (23 coeffs), Clenshaw.
    // P(u) approximates h(t) = (lgamma(t) + log(t)) / t (smooth on [0, 0.66]).
    constexpr double MID_S = 0x1.5200000000000p-2;
    constexpr double HW_S = 0x1.5200000000000p-2;
    constexpr double CHEB_S[23] = {
        -0x1.6ab12743e97f5p-2,  0x1.ac5810962c7c9p-3,   -0x1.9fc950c97fe08p-7,
        0x1.182a0ad81ff45p-10,  -0x1.ae37b402356cbp-14, 0x1.617d9d1a7276fp-17,
        -0x1.2e73d84dd6c00p-20, 0x1.09bfc3957e271p-23,  -0x1.dbcc653579713p-27,
        0x1.afdf73641ecc6p-30,  -0x1.8c3941844323bp-33, 0x1.6ea57fc6795d5p-36,
        -0x1.55ad18a495c31p-39, 0x1.404bbf9a485bcp-42,  -0x1.2dc653db9bc94p-45,
        0x1.1d92560a2236bp-48,  -0x1.0f46c20905858p-51, 0x1.02925450b709dp-54,
        -0x1.ee6a24e4dcbd9p-58, 0x1.d9f7b9780e3c5p-61,  -0x1.c77a4c49dbb90p-64,
        0x1.b6972ac7c1946p-67,  -0x1.a12b8ba3e9c77p-70};
    double u = (abs_xd - MID_S) / HW_S;
    double u2 = 2.0 * u;
    double b1 = CHEB_S[22], b2 = 0.0;
    for (int k = 21; k >= 1; --k) {
      double b0 = fputil::multiply_add(u2, b1, CHEB_S[k] - b2);
      b2 = b1;
      b1 = b0;
    }
    double poly_h = fputil::multiply_add(u, b1, CHEB_S[0] - b2);
    if (xbits.is_neg()) {
      // log(pi) - log(sin(pi*x)) - poly_val, fused via FMA to cut roundings.
      double neg_log_sin = -math::log(lg_sinpi(abs_xd));
      lgamma_val = fputil::multiply_add(-abs_xd, poly_h,
                                        0x1.250d048e7a1bdp+0 + neg_log_sin);
    } else {
      // poly_val - log(abs_xd) with single rounding via FMA.
      lgamma_val = fputil::multiply_add(abs_xd, poly_h, -math::log(abs_xd));
    }
  } else if (x_abs < 0x3f800000u) {
    // M1: t in [0.66015625, 1.0). lgamma(t) = (t-1) * P_M1(u). Degree 22.
    constexpr double MID_M1 = 0x1.a900000000000p-1;
    constexpr double HW_M1 = 0x1.5c00000000000p-3;
    constexpr double CHEB_M1[23] = {
        -0x1.7a316a5bdc767p-1,  0x1.5b2bb19b764c1p-3,   -0x1.1b0b304434f9bp-7,
        0x1.3ed224837f481p-11,  -0x1.96032c59a8abfp-15, 0x1.1312af063c88ep-18,
        -0x1.82beb75121dd4p-22, 0x1.16a1dfdaac960p-25,  -0x1.98a1140d9aeeap-29,
        0x1.2faea3c424724p-32,  -0x1.c8307861441f9p-36, 0x1.599dab4beb82dp-39,
        -0x1.07bc40924e4c5p-42, 0x1.94f8259fea698p-46,  -0x1.388d3c796c827p-49,
        0x1.e4a6b23ad71bcp-53,  -0x1.7942bc85950eap-56, 0x1.26b4a27f3a966p-59,
        -0x1.cde22532dd461p-63, 0x1.6af8963399134p-66,  -0x1.1df7a1df10080p-69,
        0x1.c398dfa01e0e3p-73,  -0x1.61e18974cb9b3p-76};
    double u = (abs_xd - MID_M1) / HW_M1;
    double u2 = 2.0 * u;
    double b1 = CHEB_M1[22], b2 = 0.0;
    for (int k = 21; k >= 1; --k) {
      double b0 = fputil::multiply_add(u2, b1, CHEB_M1[k] - b2);
      b2 = b1;
      b1 = b0;
    }
    double poly = fputil::multiply_add(u, b1, CHEB_M1[0] - b2);
    lgamma_val = (abs_xd - 1.0) * poly;
    if (xbits.is_neg()) {
      double frac_x = xd - fputil::floor(xd);
      lgamma_val = 0x1.250d048e7a1bdp+0 - lgamma_val;
      lgamma_val -= lg_ln(lg_sinpi(frac_x) * abs_xd);
    }
  } else if (x_abs < 0x40000000u) {
    // M2: t in [1.0, 2.0). lgamma(t) = (t-1)*(t-2) * P_M2(u). Degree 22 —
    // hard cases near the lgamma minimum (~1.46) demand larger headroom.
    constexpr double MID_M2 = 0x1.8000000000000p+0;
    constexpr double HW_M2 = 0x1.0000000000000p-1;
    constexpr double CHEB_M2[23] = {
        0x1.f73598c73fb27p-2,   -0x1.37c37bb231109p-4,  0x1.144f77fa9d4e5p-7,
        -0x1.1afb9900d693bp-10, 0x1.387dc81da3774p-13,  -0x1.68eb0dd19f30ep-16,
        0x1.ad3824063f31fp-19,  -0x1.047929385160cp-21, 0x1.40e5e09ac458ap-24,
        -0x1.8fe38a574f8d3p-27, 0x1.f6dc9c81f711ap-30,  -0x1.3e84fe8ed90fdp-32,
        0x1.96005b0ce149ap-35,  -0x1.041c83c9d7060p-37, 0x1.4ecb51ecebfc7p-40,
        -0x1.b09e091ec416cp-43, 0x1.187b631152304p-45,  -0x1.6cd0f97db61f7p-48,
        0x1.dbd1c1a5b72f3p-51,  -0x1.371247796da23p-53, 0x1.97a45af7c3ef7p-56,
        -0x1.0b7547cbef874p-58, 0x1.5690ac55f96fcp-61};
    double u = (abs_xd - MID_M2) / HW_M2;
    double u2 = 2.0 * u;
    double b1 = CHEB_M2[22], b2 = 0.0;
    for (int k = 21; k >= 1; --k) {
      double b0 = fputil::multiply_add(u2, b1, CHEB_M2[k] - b2);
      b2 = b1;
      b1 = b0;
    }
    double poly = fputil::multiply_add(u, b1, CHEB_M2[0] - b2);
    lgamma_val = (abs_xd - 1.0) * (abs_xd - 2.0) * poly;
    if (xbits.is_neg()) {
      double frac_x = xd - fputil::floor(xd);
      lgamma_val = 0x1.250d048e7a1bdp+0 - lgamma_val;
      lgamma_val -= lg_ln(lg_sinpi(frac_x) * abs_xd);
    }
  } else if (x_abs < 0x4057e000u) {
    // M3: t in [2.0, 3.373046875). lgamma(t) = (t-2) * P_M3(u). Degree 22.
    constexpr double MID_M3 = 0x1.57e0000000000p+1;
    constexpr double HW_M3 = 0x1.5f80000000000p-1;
    constexpr double CHEB_M3[23] = {
        0x1.377590a2d969ep-1,   0x1.66c0d826233d6p-3,   -0x1.3868a8220c21ap-7,
        0x1.89b1790a524e9p-11,  -0x1.22a7465347af0p-14, 0x1.d450a3bd37b6ep-18,
        -0x1.8e6073ea06924p-21, 0x1.5f7f615a0f66ap-24,  -0x1.3e43e37b75ba3p-27,
        0x1.25b86af37bf32p-30,  -0x1.13066b68132aep-33, 0x1.0472e866f0a58p-36,
        -0x1.f1c3c21853665p-40, 0x1.df2b520b24b05p-43,  -0x1.d016652966750p-46,
        0x1.c3ca97d4de8a2p-49,  -0x1.b9c00d6288552p-52, 0x1.b1919741c4e8ep-55,
        -0x1.aaf26c5eb3ff4p-58, 0x1.a5a74c8e09064p-61,  -0x1.a1817398c1204p-64,
        0x1.9e43a9d4a06e1p-67,  -0x1.95b4706f5cbbdp-70};
    double u = (abs_xd - MID_M3) / HW_M3;
    double u2 = 2.0 * u;
    double b1 = CHEB_M3[22], b2 = 0.0;
    for (int k = 21; k >= 1; --k) {
      double b0 = fputil::multiply_add(u2, b1, CHEB_M3[k] - b2);
      b2 = b1;
      b1 = b0;
    }
    double poly = fputil::multiply_add(u, b1, CHEB_M3[0] - b2);
    lgamma_val = (abs_xd - 2.0) * poly;
    if (xbits.is_neg()) {
      // Near the regular lgamma zero at x ~= -2.7475: subtractive cancellation
      // in the reflection formula kills precision. Use a Taylor expansion
      // centered at the zero. Range bits in (0x402f95c2, 0x40301b93).
      // Coefficients adopted from CORE-MATH (Sibidanov, 2023).
      if (LIBC_UNLIKELY(x_abs > 0x402f95c2u && x_abs < 0x40301b93u)) {
        double h = (xd + 0x1.5fb410a1bd901p+1) - 0x1.a19a96d2e6f85p-54;
        constexpr double C[8] = {-0x1.ea12da904b18cp+0,  0x1.3267f3c265a54p+3,
                                 -0x1.4185ac30cadb3p+4,  0x1.f504accc3f2e4p+5,
                                 -0x1.8588444c679b4p+7,  0x1.43740491dc22p+9,
                                 -0x1.12400ea23f9e6p+11, 0x1.dac829f365795p+12};
        double h2 = h * h, h4 = h2 * h2;
        double p01 = fputil::multiply_add(h, C[1], C[0]);
        double p23 = fputil::multiply_add(h, C[3], C[2]);
        double p45 = fputil::multiply_add(h, C[5], C[4]);
        double p67 = fputil::multiply_add(h, C[7], C[6]);
        double p03 = fputil::multiply_add(h2, p23, p01);
        double p47 = fputil::multiply_add(h2, p67, p45);
        lgamma_val = h * fputil::multiply_add(h4, p47, p03);
      } else if (LIBC_UNLIKELY(x_abs > 0x401ceccbu && x_abs < 0x401d95cau)) {
        // Near the regular lgamma zero at x ~= -2.3614: same issue
        double h = (xd + 0x1.3a7fc9600f86cp+1) + 0x1.55f64f98af8dp-55;
        constexpr double C[7] = {0x1.83fe966af535fp+0, 0x1.36eebb002f61ap+2,
                                 0x1.694a60589a0b3p+0, 0x1.1718d7aedb0b5p+3,
                                 0x1.733a045eca0d3p+2, 0x1.8d4297421205bp+4,
                                 0x1.7feea5fb29965p+4};
        double h2 = h * h, h4 = h2 * h2;
        double p01 = fputil::multiply_add(h, C[1], C[0]);
        double p23 = fputil::multiply_add(h, C[3], C[2]);
        double p45 = fputil::multiply_add(h, C[5], C[4]);
        double p46 = fputil::multiply_add(h2, C[6], p45);
        double p03 = fputil::multiply_add(h2, p23, p01);
        lgamma_val = h * fputil::multiply_add(h4, p46, p03);
      } else if (LIBC_UNLIKELY(x_abs > 0x40492009u && x_abs < 0x404940efu)) {
        // Near the regular lgamma zero at x ~= -3.1431: same issue
        double h = (xd + 0x1.9260dbc9e59afp+1) + 0x1.f717cd335a7b3p-53;
        constexpr double C[7] = {0x1.f20a65f2fac55p+2,  0x1.9d4d297715105p+4,
                                 0x1.c1137124d5b21p+6,  0x1.267203d24de38p+9,
                                 0x1.99a63399a0b44p+11, 0x1.2941214faaf0cp+14,
                                 0x1.bb912c0c9cdd1p+16};
        double h2 = h * h, h4 = h2 * h2;
        double p01 = fputil::multiply_add(h, C[1], C[0]);
        double p23 = fputil::multiply_add(h, C[3], C[2]);
        double p45 = fputil::multiply_add(h, C[5], C[4]);
        double p46 = fputil::multiply_add(h2, C[6], p45);
        double p03 = fputil::multiply_add(h2, p23, p01);
        lgamma_val = h * fputil::multiply_add(h4, p46, p03);
      } else {
        double frac_x = xd - fputil::floor(xd);
        lgamma_val = 0x1.250d048e7a1bdp+0 - lgamma_val;
        lgamma_val -= lg_ln(lg_sinpi(frac_x) * abs_xd);
      }
    }
  } else {
    // Large: |x| >= 3.373046875. Stirling + Bernoulli correction.
    // lgamma(x) = (x-0.5)*log(x) - x + log(2*pi)/2 + (1/x)*P(1/x^2)
    //          = (x-0.5)*(log(x)-1) + STIR_CONST + (1/x)*P(1/x^2)
    // STIR_CONST = log(2*pi)/2 - 0.5.
    //
    //   log(x) = e*log(2) + log(m)        where x = 2^e * m, m in [1, 2)
    //          = e*LOG2_HI + (e*LOG2_LO + log(m))
    // LOG2_HI has 44 sig bits, so e*LOG2_HI is exact for |e| < 512.
    // For huge positive x, lgamma(x) overflows float. Use a linear
    // approximation in double that maps to the correct Inf/max_normal.
    if (LIBC_UNLIKELY(!xbits.is_neg() && x >= 0x1.895f1cp+121f)) {
      double r = fputil::multiply_add(xd, 0x1.4d3398p+6, 0x1.10f35ep+103);
      float result = fputil::cast<float>(r);
      if (FPBits(result).is_inf()) {
        fputil::raise_except_if_required(FE_OVERFLOW | FE_INEXACT);
        fputil::set_errno_if_required(ERANGE);
      }
      return result;
    }

    constexpr double LOG2_HI = 0x1.62e42fefa3800p-1;
    constexpr double LOG2_LO = 0x1.ef00000000000p-45;
    using DPBits = fputil::FPBits<double>;
    DPBits abs_bits(abs_xd);
    int e = static_cast<int>(abs_bits.get_biased_exponent()) - 0x3ff;
    // m in [1, 2)
    DPBits m_bits = abs_bits;
    m_bits.set_biased_exponent(0x3ff);
    double m = m_bits.get_val();
    double log_m = math::log(m);
    double e_d = static_cast<double>(e);
    double log_hi = e_d * LOG2_HI; // exact
    double log_lo = fputil::multiply_add(e_d, LOG2_LO, log_m);
    fputil::DoubleDouble log_x = fputil::exact_add(log_hi, log_lo);

    double xm = abs_xd - 0.5;
    // (xm) * log_x
    fputil::DoubleDouble prod = fputil::exact_mult(xm, log_x.hi);
    prod.lo = fputil::multiply_add(xm, log_x.lo, prod.lo);

    // result_main = (xm * log_x) - xm + STIR_CONST
    fputil::DoubleDouble diff = fputil::exact_add(prod.hi, -xm);
    diff.lo += prod.lo;
    lgamma_val = diff.hi + (diff.lo + 0x1.acfe390c97d69p-2);

    double inv_x = 1.0 / abs_xd;
    double inv_x2 = inv_x * inv_x;

    if (x_abs > 0x4b989680u) {
      // |x| > 2e7 -> just 1/(12x). Next Bernoulli term is < 2^-80.
      lgamma_val += inv_x * 0x1.5555555555555p-4;
    } else if (x_abs > 0x44fa0000u) {
      // |x| > 2000 -> 2-term BERN2.
      constexpr double BERN2[2] = {0x1.5555555555555p-4, -0x1.6c16bfb7c65a8p-9};
      lgamma_val += inv_x * fputil::multiply_add(inv_x2, BERN2[1], BERN2[0]);
    } else if (x_abs > 0x42920000u) {
      // |x| > 73 -> 4-term BERN4.
      constexpr double BERN4[4] = {0x1.5555555555555p-4, -0x1.6c16c16c15f75p-9,
                                   0x1.a01a00593b36fp-11,
                                   -0x1.37e91273668efp-11};
      double inv_x4 = inv_x2 * inv_x2;
      double p01 = fputil::multiply_add(inv_x2, BERN4[1], BERN4[0]);
      double p23 = fputil::multiply_add(inv_x2, BERN4[3], BERN4[2]);
      lgamma_val += inv_x * fputil::multiply_add(inv_x4, p23, p01);
    } else {
      // |x| in (3.373, 73]: degree-9 Chebyshev poly in s = 1/x^2 for the
      // function h(s) = stir_resid(1/sqrt(s)) * (1/sqrt(s)), evaluated by
      // Clenshaw. Then correction = h(s) * (1/x). Max abs error ~5.5e-18.
      constexpr double CHEB_B10_MID = 0x1.68c7762f8b34ep-5;
      constexpr double CHEB_B10_HW = 0x1.673ded08c6ba2p-5;
      constexpr double CHEB_B10[10] = {
          0x1.54d759afaa7e1p-4,  -0x1.f2c7cca18bb6bp-14,
          0x1.7601881c38683p-21, -0x1.5a6f789402bbcp-27,
          0x1.198036ed27e9bp-32, -0x1.5459d5559def8p-37,
          0x1.15c4fb498d074p-41, -0x1.1f1b0949ee034p-45,
          0x1.6759cacff87f6p-49, -0x1.046d1e7698673p-52};
      double u = (inv_x2 - CHEB_B10_MID) / CHEB_B10_HW;
      double u2 = 2.0 * u;
      double b1 = CHEB_B10[9], b2 = 0.0;
      for (int k = 8; k >= 1; --k) {
        double b0 = fputil::multiply_add(u2, b1, CHEB_B10[k] - b2);
        b2 = b1;
        b1 = b0;
      }
      double poly = fputil::multiply_add(u, b1, CHEB_B10[0] - b2);
      lgamma_val += inv_x * poly;
    }

    if (xbits.is_neg()) {
      // Reflection: lgamma(x) = log(pi) - lgamma(|x|) -
      // log(|x|*|sin(pi*frac_x)|)
      double frac_x = xd - fputil::floor(xd);
      lgamma_val = 0x1.250d048e7a1bdp+0 - lgamma_val;
      lgamma_val -= lg_ln(lg_sinpi(frac_x) * abs_xd);
    }
  }

  float result = fputil::cast<float>(lgamma_val);
  if (LIBC_UNLIKELY(FPBits(result).is_inf())) {
    fputil::raise_except_if_required(FE_OVERFLOW | FE_INEXACT);
    fputil::set_errno_if_required(ERANGE);
  }
  return result;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_LGAMMAF_H
