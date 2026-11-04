//===-- Double-precision tgamma function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMA_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMA_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double tgamma(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);

  if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    // tgamma(-inf) = NaN with domain error
    if (xbits.is_neg()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    // tgamma(+inf) = +inf
    return x;
  }

  // Gamma has a pole at 0, with sign following the input
  if (LIBC_UNLIKELY(x == 0.0)) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_DIVBYZERO);
    return FPBits::inf(xbits.sign()).get_val();
  }

  // Range 1: Extremely small x, |x| < 2 ** -53
  // We just approximate Gamma(x) as 1 / x
  if (LIBC_UNLIKELY(xbits.abs().uintval() < FPBits(0x1.0p-53).uintval())) {
    double r = 1.0 / x;

    // Sufficiently tiny |x| pushes 1 / x past DBL_MAX leading to overflow
    if (LIBC_UNLIKELY(FPBits(r).is_inf())) {
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_OVERFLOW);
    }

    // Gamma(x) is never exactly 1 / x here
    fputil::raise_except_if_required(FE_INEXACT);
    return r;
  }

  // tgamma(x) > DBL_MAX for x >= 0x1.573fae561f648p+7
  // Source: https://members.loria.fr/PZimmermann/papers/gamma.pdf
  if (LIBC_UNLIKELY(x >= 0x1.573fae561f648p+7)) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_OVERFLOW);
    return FPBits::inf().get_val();
  }

  // Gamma has poles at all negative integers
  if (LIBC_UNLIKELY(x < 0.0 && fputil::nearest_integer(x) == x)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // Range 2: x is a positive integer in [1, 171]
  // FACTORIALS[x] = x!, correctly rounded to nearest double
  // tgamma(x) = FACTORIALS[x - 1]
  // x = 0..22 exact, x = 23..170 correctly-rounded
  // 170! is the largest factorial that fits under DBL_MAX
  // Generated using a simple Sollya for-loop
  static constexpr double FACTORIALS[171] = {
      0x1.0000000000000p+0,    0x1.0000000000000p+0,
      0x1.0000000000000p+1,    0x1.8000000000000p+2,
      0x1.8000000000000p+4,    0x1.e000000000000p+6,
      0x1.6800000000000p+9,    0x1.3b00000000000p+12,
      0x1.3b00000000000p+15,   0x1.6260000000000p+18,
      0x1.baf8000000000p+21,   0x1.308a800000000p+25,
      0x1.c8cfc00000000p+28,   0x1.7328cc0000000p+32,
      0x1.44c3b28000000p+36,   0x1.3077775800000p+40,
      0x1.3077775800000p+44,   0x1.437eeecd80000p+48,
      0x1.6beecca730000p+52,   0x1.b02b930689000p+56,
      0x1.0e1b3be415a00p+61,   0x1.6283be9b5c620p+65,
      0x1.e77526159f06cp+69,   0x1.5e5c335f8a4cep+74,
      0x1.06c52687a7b9ap+79,   0x1.9a940c33f6121p+83,
      0x1.4d9849ea37eebp+88,   0x1.19787e5d9f316p+93,
      0x1.ec92dd23d6967p+97,   0x1.be6518687a785p+102,
      0x1.a27ec6e1f2d0dp+107,  0x1.956ad0aae33a4p+112,
      0x1.956ad0aae33a4p+117,  0x1.a21627303a541p+122,
      0x1.bc3789a33df96p+127,  0x1.e5dcbe8a8bc8cp+132,
      0x1.114c2b2deea0fp+138,  0x1.3c0011ed1bea1p+143,
      0x1.774015499125fp+148,  0x1.c95619f1a8e64p+153,
      0x1.1dd5d037098fep+159,  0x1.6e39f2c684406p+164,
      0x1.e0ac0ea48d948p+169,  0x1.42f399d68f1fcp+175,
      0x1.bc0ef38704cbbp+180,  0x1.383a833aef5f3p+186,
      0x1.c0d41ca4b818ep+191,  0x1.499bc508f7324p+197,
      0x1.ee69a78d72cb6p+202,  0x1.7a88e4484be3bp+208,
      0x1.27baf2587b49ep+214,  0x1.d751f23d047dcp+219,
      0x1.7ef294d193a63p+225,  0x1.3d20e33d8e45ap+231,
      0x1.0b93bfbbf00acp+237,  0x1.cbe5f18b04928p+242,
      0x1.92693359a4003p+248,  0x1.6665b1bbd6102p+254,
      0x1.44cc291239feap+260,  0x1.2b6c35dccd76cp+266,
      0x1.18b5727f009f5p+272,  0x1.0b8cf1210c97ep+278,
      0x1.0330899804332p+284,  0x1.fe478ee34844ap+289,
      0x1.fe478ee34844ap+295,  0x1.0320568f6ab2ep+302,
      0x1.0b395943e6087p+308,  0x1.17c0097314d0dp+314,
      0x1.293c0a0a461dep+320,  0x1.4074bad313983p+326,
      0x1.5e7fac56dd6e8p+332,  0x1.84d5a3305da69p+338,
      0x1.b5705796695b6p+344,  0x1.f2f423e7902c4p+350,
      0x1.207524c1df599p+357,  0x1.5209471331bd0p+363,
      0x1.916b0466cb107p+369,  0x1.e2f4c14bac4fcp+375,
      0x1.264d25ca1d009p+382,  0x1.6b473aa57bcccp+388,
      0x1.c619094edabffp+394,  0x1.1f5bd7e3e66d7p+401,
      0x1.702dac9bff3c4p+407,  0x1.dd7b3bda4f022p+413,
      0x1.3958df4743d96p+420,  0x1.a02a088aa61cbp+426,
      0x1.179c3dbd279b5p+433,  0x1.7c1863ed21d72p+439,
      0x1.0550c4b30743ep+446,  0x1.6b645188f61a6p+452,
      0x1.ff0512a89a152p+458,  0x1.6b4d9b43dd8b0p+465,
      0x1.051fc798c73bfp+472,  0x1.7b722e0a01831p+478,
      0x1.16a7d9cf591c4p+485,  0x1.9da1274fc845fp+491,
      0x1.3638dd7bd6347p+498,  0x1.d62e2fafb0a78p+504,
      0x1.67fb5c8283404p+511,  0x1.166c698cf183bp+518,
      0x1.b30964ec395dcp+524,  0x1.574569a265440p+531,
      0x1.118b502d68b23p+538,  0x1.b83c3509147ecp+544,
      0x1.65b0eb1760a70p+551,  0x1.256b20d92d490p+558,
      0x1.e5f96e67b300ep+564,  0x1.963e824aafa2cp+571,
      0x1.56c4bdef04315p+578,  0x1.23e389bd89920p+585,
      0x1.f5af14bdc472fp+591,  0x1.b30dd3fc905bap+598,
      0x1.7cac197cfe503p+605,  0x1.500fee805882dp+612,
      0x1.2b4e306a4ed48p+619,  0x1.0ce83f7f82d2fp+626,
      0x1.e764f3171d1e4p+632,  0x1.bd824633209dbp+639,
      0x1.9ab418b722116p+646,  0x1.7dd36efa41ac2p+653,
      0x1.65f6380a9d916p+660,  0x1.5262c0fa08f37p+667,
      0x1.42861fee50880p+674,  0x1.35ece2af0162bp+681,
      0x1.2c3d7b998957ap+688,  0x1.25340ab3f01f9p+695,
      0x1.209f3a89205f1p+702,  0x1.1e5dfc140e1e5p+709,
      0x1.1e5dfc140e1e5p+716,  0x1.209ab80c363a9p+723,
      0x1.251d22ec67138p+730,  0x1.2bfbd1bdf17dfp+737,
      0x1.355bb04be109ep+744,  0x1.4171452ed7d44p+751,
      0x1.5082946d09f23p+758,  0x1.62e9b88b007d7p+765,
      0x1.79185413b0855p+772,  0x1.939c09fd12eebp+779,
      0x1.b3243ac4d8695p+786,  0x1.d88957d1c3026p+793,
      0x1.026b1c06b6a55p+801,  0x1.1ca9fcdf65321p+808,
      0x1.3bcc9487d4439p+815,  0x1.60ce8defbf238p+822,
      0x1.8ce85fadb707ep+829,  0x1.c19f3c62c956fp+836,
      0x1.006cd07056d39p+844,  0x1.267cf76103b70p+851,
      0x1.54807e082c4b9p+858,  0x1.8c5d92b583900p+865,
      0x1.d07da7ecb62ccp+872,  0x1.11fa1e0c9f746p+880,
      0x1.455903aefd5a3p+887,  0x1.84e466672ad5dp+894,
      0x1.d3e2cb341f894p+901,  0x1.1b4a51088f182p+909,
      0x1.594292c26e656p+916,  0x1.a77ba8027b686p+923,
      0x1.055e51b1882a7p+931,  0x1.44ab297a8724bp+938,
      0x1.95d5f3d928edep+945,  0x1.fe771cb7257b3p+952,
      0x1.4307602be5b7fp+960,  0x1.9b5b6477e6884p+967,
      0x1.07868c5ccfaf4p+975,  0x1.53b370efa3b7fp+982,
      0x1.b88cb676c8529p+989,  0x1.1f63cb077cadep+997,
      0x1.7932fa79d3a43p+1004, 0x1.f2054eb4d96ecp+1011,
      0x1.4ab7864418639p+1019};

  // It is safe to cast to int at this point because any integer
  // reaching this part of the code is at most 171 because of the
  // overflow check above.
  if (LIBC_UNLIKELY(x > 0.0 && fputil::nearest_integer(x) == x)) {
    return FACTORIALS[static_cast<int>(x) - 1];
  }

  // TODO: Implement tgamma for the remaining input ranges
  return 0.0;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMA_H
