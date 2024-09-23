//===-- Half-precision sinpif function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define M_PI 3.1415925f

#include "src/math/sinpif16.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

// TODO: Should probably create a new file; sincospif16_utils.h
// To store the following helper functions and constants.
// I'd defer to @lntue for suggestions regarding that

// HELPER_START
namespace LIBC_NAMESPACE_DECL {

constexpr float PI_OVER_32 = M_PI / 32;

// In Sollya generate 10 coeffecients for a degree-9 chebyshev polynomial
// approximating the sine function in [-pi / 32, pi / 32] with the following
// commands:
// > prec=23;
// > TL = chebyshevform(sin(x), 9, [-pi / 32, pi / 32]);
// > TL[0];
const float SIN_COEFF[10] = {
    0x1.801p-27, 0x1.000078p0, -0x1.7e98p-14, -0x1.6bf4p-3, 0x1.95ccp-5,
    0x1.1baep2,  -0x1.030ap3,  -0x1.3dap9,    0x1.98e4p8,   0x1.d3d8p14};

// In Sollya generate 10 coefficients for a degree-9 chebyshev polynomial
// approximating the sine function in [-pi/32, pi/32] with the following
// commands:
// > prec = 23;
// > TL = chebyshevform(cos(x), 9, [-pi / 32, pi / 32]);
// > TL[0];
const float COS_COEFF[10] = {
    0x1.00001p0, -0x1.48p-17, -0x1.01259cp-1, -0x1.17fp-6, 0x1.283p0,
    0x1.5d1p3,   -0x1.6278p7, -0x1.c23p10,    0x1.1444p13, 0x1.5fcp16};

// Lookup table for sin(k * pi / 32) with k = 0, ..., 63.
// Table is generated with Sollya as follows:
// > display = hexadecimmal;
// > prec = 23;
// > for k from 0 to 63 do {sin(k * pi/32);};

const float SIN_K_PI_OVER_32[64] = {0,
                                    0x1.917a6cp-4,
                                    0x1.8f8b84p-3,
                                    0x1.294064p-2,
                                    0x1.87de2cp-2,
                                    0x1.e2b5d4p-2,
                                    0x1.1c73b4p-1,
                                    0x1.44cf34p-1,
                                    0x1.6a09e8p-1,
                                    0x1.8bc808p-1,
                                    0x1.a9b664p-1,
                                    0x1.c38b3p-1,
                                    0x1.d906bcp-1,
                                    0x1.e9f414p-1,
                                    0x1.f6297cp-1,
                                    0x1.fd88dcp-1,
                                    0x1p0,
                                    0x1.fd88dcp-1,
                                    0x1.f6297cp-1,
                                    0x1.e9f414p-1,
                                    0x1.d906bcp-1,
                                    0x1.c38b3p-1,
                                    0x1.a9b664p-1,
                                    0x1.8bc808p-1,
                                    0x1.6a09e8p-1,
                                    0x1.44cf34p-1,
                                    0x1.1c73b4p-1,
                                    0x1.e2b5d4p-2,
                                    0x1.87de2cp-2,
                                    0x1.294064p-2,
                                    0x1.8f8b84p-3,
                                    0x1.917a6cp-4,
                                    0,
                                    -0x1.917a6cp-4,
                                    -0x1.8f8b84p-3,
                                    -0x1.294064p-2,
                                    -0x1.87de2cp-2,
                                    -0x1.e2b5d4p-2,
                                    -0x1.1c73b4p-1,
                                    -0x1.44cf34p-1,
                                    -0x1.6a09e8p-1,
                                    -0x1.8bc808p-1,
                                    -0x1.a9b664p-1,
                                    -0x1.c38b3p-1,
                                    -0x1.d906bcp-1,
                                    -0x1.e9f414p-1,
                                    -0x1.f6297cp-1,
                                    -0x1.fd88dcp-1,
                                    -0x1p0,
                                    -0x1.fd88dcp-1,
                                    -0x1.f6297cp-1,
                                    -0x1.e9f414p-1,
                                    -0x1.d906bcp-1,
                                    -0x1.c38b3p-1,
                                    -0x1.a9b664p-1,
                                    -0x1.8bc808p-1,
                                    -0x1.6a09e8p-1,
                                    -0x1.44cf34p-1,
                                    -0x1.1c73b4p-1,
                                    -0x1.e2b5d4p-2,
                                    -0x1.87de2cp-2,
                                    -0x1.294064p-2,
                                    -0x1.8f8b84p-3,
                                    -0x1.917a6cp-4};

// horner's algorithm to accurately and efficiently evaluate a degree-9
// polynomial iteratively
float horners(float x, const float COEFF[10]) {
  float b8 = fputil::multiply_add<float>(COEFF[9], x, COEFF[8]);
  float b7 = fputil::multiply_add<float>(b8, x, COEFF[7]);
  float b6 = fputil::multiply_add<float>(b7, x, COEFF[6]);
  float b5 = fputil::multiply_add<float>(b6, x, COEFF[5]);
  float b4 = fputil::multiply_add<float>(b5, x, COEFF[4]);
  float b3 = fputil::multiply_add<float>(b4, x, COEFF[3]);
  float b2 = fputil::multiply_add<float>(b3, x, COEFF[2]);
  float b1 = fputil::multiply_add<float>(b2, x, COEFF[1]);
  return fputil::multiply_add<float>(b1, x, COEFF[0]);
}

float range_reduction(float x, float &y) {
  float kf = fputil::nearest_integer(x * 32);
  y = fputil::multiply_add<float>(x, 32.0, -kf);

  return static_cast<int32_t>(kf);
}
// HELPER_END

LLVM_LIBC_FUNCTION(float16, sinpif16, (float16 x)) {
  using FPBits = typename fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  // Range reduction:
  // For |x| > 1/32, we perform range reduction as follows:
  // Find k and y such that:
  //   x = (k + y) * 1/32
  //   k is an integer
  //   |y| < 0.5
  //
  // This is done by performing:
  //   k = round(x * 32)
  //   y = x * 32 - k
  //
  // Once k and y are computed, we then deduce the answer by the sine of sum
  // formula:
  //   sin(x * pi) = sin((k + y) * pi/32)
  //           = sin(k * pi/32) * cos(y * pi/32) + sin (y * pi/32) * cos (k *
  //           pi/32)
  // The values of sin(k * pi/32) and cos (k * pi/32) for k = 0...63 are
  // precomputed and stored using a vector of 64 single precision floats. sin(y
  // * pi/32) and cos(y * pi/32) are computed using degree-9 chebyshev
  // polynomials generated by Sollya.

  float f32 = x;
  float y;
  int32_t k = range_reduction(f32, y);

  float sin_k = SIN_K_PI_OVER_32[k & 63];
  float cos_k = SIN_K_PI_OVER_32[(k + 16) & 63];

  float cos_y, sin_y;
  if (y == 0) {
    cos_y = 1;
    sin_y = 0;
  } else {
    cos_y = horners(y * PI_OVER_32, COS_COEFF);
    sin_y = horners(y * PI_OVER_32, SIN_COEFF);
  }

  return static_cast<float16>(fputil::multiply_add(
      sin_k, cos_y, fputil::multiply_add(sin_y, cos_k, 0)));
}
} // namespace LIBC_NAMESPACE_DECL
