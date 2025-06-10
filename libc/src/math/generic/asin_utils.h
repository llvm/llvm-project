//===-- Collection of utils for asin/acos -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_ASIN_UTILS_H
#define LLVM_LIBC_SRC_MATH_GENERIC_ASIN_UTILS_H

#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/integer_literals.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

using DoubleDouble = fputil::DoubleDouble;
using Float128 = fputil::DyadicFloat<128>;

constexpr DoubleDouble PI = {0x1.1a62633145c07p-53, 0x1.921fb54442d18p1};

constexpr DoubleDouble PI_OVER_TWO = {0x1.1a62633145c07p-54,
                                      0x1.921fb54442d18p0};

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS

// When correct rounding is not needed, we use a degree-22 minimax polynomial to
// approximate asin(x)/x on [0, 0.5] using Sollya with:
// > P = fpminimax(asin(x)/x, [|0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22|],
//                 [|1, D...|], [0, 0.5]);
// > dirtyinfnorm(asin(x)/x - P, [0, 0.5]);
// 0x1.1a71ef0a0f26a9fb7ed7e41dee788b13d1770db3dp-52

constexpr double ASIN_COEFFS[12] = {
    0x1.0000000000000p0,  0x1.5555555556dcfp-3,  0x1.3333333082e11p-4,
    0x1.6db6dd14099edp-5, 0x1.f1c69b35bf81fp-6,  0x1.6e97194225a67p-6,
    0x1.1babddb82ce12p-6, 0x1.d55bd078600d6p-7,  0x1.33328959e63d6p-7,
    0x1.2b5993bda1d9bp-6, -0x1.806aff270bf25p-7, 0x1.02614e5ed3936p-5,
};

LIBC_INLINE double asin_eval(double u) {
  double u2 = u * u;
  double c0 = fputil::multiply_add(u, ASIN_COEFFS[1], ASIN_COEFFS[0]);
  double c1 = fputil::multiply_add(u, ASIN_COEFFS[3], ASIN_COEFFS[2]);
  double c2 = fputil::multiply_add(u, ASIN_COEFFS[5], ASIN_COEFFS[4]);
  double c3 = fputil::multiply_add(u, ASIN_COEFFS[7], ASIN_COEFFS[6]);
  double c4 = fputil::multiply_add(u, ASIN_COEFFS[9], ASIN_COEFFS[8]);
  double c5 = fputil::multiply_add(u, ASIN_COEFFS[11], ASIN_COEFFS[10]);

  double u4 = u2 * u2;
  double d0 = fputil::multiply_add(u2, c1, c0);
  double d1 = fputil::multiply_add(u2, c3, c2);
  double d2 = fputil::multiply_add(u2, c5, c4);

  return fputil::polyeval(u4, d0, d1, d2);
}

#else

// The Taylor expansion of asin(x) around 0 is:
//   asin(x) = x + x^3/6 + 3x^5/40 + ...
//           ~ x * P(x^2).
// Let u = x^2, then P(x^2) = P(u), and |x| = sqrt(u).  Note that when
// |x| <= 0.5, we have |u| <= 0.25.
// We approximate P(u) by breaking it down by performing range reduction mod
//   2^-5 = 1/32.
// So for:
//   k = round(u * 32),
//   y = u - k/32,
// we have that:
//   x = sqrt(u) = sqrt(k/32 + y),
//   |y| <= 2^-5 = 1/32,
// and:
//   P(u) = P(k/32 + y) = Q_k(y).
// Hence :
//   asin(x) = sqrt(k/32 + y) * Q_k(y),
// Or equivalently:
//   Q_k(y) = asin(sqrt(k/32 + y)) / sqrt(k/32 + y).
// We generate the coefficients of Q_k by Sollya as following:
// > procedure ASIN_APPROX(N, Deg) {
//     abs_error = 0;
//     rel_error = 0;
//     deg = [||];
//     for i from 2 to Deg do deg = deg :. i;
//     for i from 1 to N/4 do {
//       F = asin(sqrt(i/N + x))/sqrt(i/N + x);
//       T = taylor(F, 1, 0);
//       T_DD = roundcoefficients(T, [|DD...|]);
//       I = [-1/(2*N), 1/(2*N)];
//       Q = fpminimax(F, deg, [|D...|], I, T_DD);
//       abs_err = dirtyinfnorm(F - Q, I);
//       rel_err = dirtyinfnorm((F - Q)/x^2, I);
//       if (abs_err > abs_error) then abs_error = abs_err;
//       if (rel_err > rel_error) then rel_error = rel_err;
//       d0 = D(coeff(Q, 0));
//       d1 = coeff(Q, 0) - d0;
//       write("{", d0, ", ", d1);
//       d0 = D(coeff(Q, 1)); d1 = coeff(Q, 1) - d0;  write(", ", d0, ", ", d1);
//       for j from 2 to Deg do {
//         write(", ", coeff(Q, j));
//       };
//       print("},");
//     };
//     print("Absolute Errors:", D(abs_error));
//     print("Relative Errors:", D(rel_error));
//  };
// > ASIN_APPROX(32, 9);
// Absolute Errors: 0x1.69837b5183654p-72
// Relative Errors: 0x1.4d7f82835bf64p-55

// For k = 0, we use the degree-18 Taylor polynomial of asin(x)/x:
//
// > P = 1 + x^2 * DD(1/6) + x^4 * D(3/40) + x^6 * D(5/112) + x^8 * D(35/1152) +
//       x^10 * D(63/2816) + x^12 * D(231/13312) + x^14 * D(143/10240) +
//       x^16 * D(6435/557056) + x^18 * D(12155/1245184);
// > dirtyinfnorm(asin(x)/x - P, [-1/64, 1/64]);
// 0x1.999075402cafp-83

constexpr double ASIN_COEFFS[9][12] = {
    {1.0, 0.0, 0x1.5555555555555p-3, 0x1.5555555555555p-57,
     0x1.3333333333333p-4, 0x1.6db6db6db6db7p-5, 0x1.f1c71c71c71c7p-6,
     0x1.6e8ba2e8ba2e9p-6, 0x1.1c4ec4ec4ec4fp-6, 0x1.c99999999999ap-7,
     0x1.7a87878787878p-7, 0x1.3fde50d79435ep-7},
    {0x1.015a397cf0f1cp0, -0x1.eebd6ccfe3ee3p-55, 0x1.5f3581be7b08bp-3,
     -0x1.5df80d0e7237dp-57, 0x1.4519ddf1ae53p-4, 0x1.8eb4b6eeb1696p-5,
     0x1.17bc85420fec8p-5, 0x1.a8e39b5dcad81p-6, 0x1.53f8df127539bp-6,
     0x1.1a485a0b0130ap-6, 0x1.e20e6e493002p-7, 0x1.a466a7030f4c9p-7},
    {0x1.02be9ce0b87cdp0, 0x1.e5d09da2e0f04p-56, 0x1.69ab5325bc359p-3,
     -0x1.92f480cfede2dp-57, 0x1.58a4c3097aab1p-4, 0x1.b3db36068dd8p-5,
     0x1.3b9482184625p-5, 0x1.eedc823765d21p-6, 0x1.98e35d756be6bp-6,
     0x1.5ea4f1b32731ap-6, 0x1.355115764148ep-6, 0x1.16a5853847c91p-6},
    {0x1.042dc6a65ffbfp0, -0x1.c7ea28dce95d1p-55, 0x1.74c4bd7412f9dp-3,
     0x1.447024c0a3c87p-58, 0x1.6e09c6d2b72b9p-4, 0x1.ddd9dcdae5315p-5,
     0x1.656f1f64058b8p-5, 0x1.21a42e4437101p-5, 0x1.eed0350b7edb2p-6,
     0x1.b6bc877e58c52p-6, 0x1.903a0872eb2a4p-6, 0x1.74da839ddd6d8p-6},
    {0x1.05a8621feb16bp0, -0x1.e5b33b1407c5fp-56, 0x1.809186c2e57ddp-3,
     -0x1.3dcb4d6069407p-60, 0x1.8587d99442dc5p-4, 0x1.06c23d1e75be3p-4,
     0x1.969024051c67dp-5, 0x1.54e4f934aacfdp-5, 0x1.2d60a732dbc9cp-5,
     0x1.149f0c046eac7p-5, 0x1.053a56dba1fbap-5, 0x1.f7face3343992p-6},
    {0x1.072f2b6f1e601p0, -0x1.2dcbb0541997p-54, 0x1.8d2397127aebap-3,
     0x1.ead0c497955fbp-57, 0x1.9f68df88da518p-4, 0x1.21ee26a5900d7p-4,
     0x1.d08e7081b53a9p-5, 0x1.938dd661713f7p-5, 0x1.71b9f299b72e6p-5,
     0x1.5fbc7d2450527p-5, 0x1.58573247ec325p-5, 0x1.585a174a6a4cep-5},
    {0x1.08c2f1d638e4cp0, 0x1.b47c159534a3dp-56, 0x1.9a8f592078624p-3,
     -0x1.ea339145b65cdp-57, 0x1.bc04165b57aabp-4, 0x1.410df5f58441dp-4,
     0x1.0ab6bdf5f8f7p-4, 0x1.e0b92eea1fce1p-5, 0x1.c9094e443a971p-5,
     0x1.c34651d64bc74p-5, 0x1.caa008d1af08p-5, 0x1.dc165bc0c4fc5p-5},
    {0x1.0a649a73e61f2p0, 0x1.74ac0d817e9c7p-55, 0x1.a8ec30dc9389p-3,
     -0x1.8ab1c0eef300cp-59, 0x1.dbc11ea95061bp-4, 0x1.64e371d661328p-4,
     0x1.33e0023b3d895p-4, 0x1.2042269c243cep-4, 0x1.1cce74bda223p-4,
     0x1.244d425572ce9p-4, 0x1.34d475c7f1e3ep-4, 0x1.4d4e653082ad3p-4},
    {0x1.0c152382d7366p0, -0x1.ee6913347c2a6p-54, 0x1.b8550d62bfb6dp-3,
     -0x1.d10aec3f116d5p-57, 0x1.ff1bde0fa3cap-4, 0x1.8e5f3ab69f6a4p-4,
     0x1.656be8b6527cep-4, 0x1.5c39755dc041ap-4, 0x1.661e6ebd40599p-4,
     0x1.7ea3dddee2a4fp-4, 0x1.a4f439abb4869p-4, 0x1.d9181c0fda658p-4},
};

// We calculate the lower part of the approximation P(u).
LIBC_INLINE DoubleDouble asin_eval(const DoubleDouble &u, unsigned &idx,
                                   double &err) {
  using fputil::multiply_add;
  // k = round(u * 32).
  double k = fputil::nearest_integer(u.hi * 0x1.0p5);
  idx = static_cast<unsigned>(k);
  // y = u - k/32.
  double y_hi = multiply_add(k, -0x1.0p-5, u.hi); // Exact
  DoubleDouble y = fputil::exact_add(y_hi, u.lo);
  double y2 = y.hi * y.hi;
  // Add double-double errors in addition to the relative errors from y2.
  err = fputil::multiply_add(err, y2, 0x1.0p-102);
  DoubleDouble c0 = fputil::quick_mult(
      y, DoubleDouble{ASIN_COEFFS[idx][3], ASIN_COEFFS[idx][2]});
  double c1 = multiply_add(y.hi, ASIN_COEFFS[idx][5], ASIN_COEFFS[idx][4]);
  double c2 = multiply_add(y.hi, ASIN_COEFFS[idx][7], ASIN_COEFFS[idx][6]);
  double c3 = multiply_add(y.hi, ASIN_COEFFS[idx][9], ASIN_COEFFS[idx][8]);
  double c4 = multiply_add(y.hi, ASIN_COEFFS[idx][11], ASIN_COEFFS[idx][10]);

  double y4 = y2 * y2;
  double d0 = multiply_add(y2, c2, c1);
  double d1 = multiply_add(y2, c4, c3);

  DoubleDouble r = fputil::exact_add(ASIN_COEFFS[idx][0], c0.hi);

  double e1 = multiply_add(y4, d1, d0);

  r.lo = multiply_add(y2, e1, ASIN_COEFFS[idx][1] + c0.lo + r.lo);

  return r;
}

// Follow the discussion above, we generate the coefficients of Q_k by Sollya as
// following:
// > procedure PRINTF128(a) {
//   write("{");
//   if (a < 0)
//     then write("Sign::NEG, ") else write("Sign::POS, ");
//   a_exp = floor(log2(a)) + 1;
//   write((a + 2 ^ a_exp) * 2 ^ -128);
//   print("},");
// };
// > verbosity = 0;
// > procedure ASIN_APPROX(N, Deg) {
//     abs_error = 0;
//     rel_error = 0;
//     for i from 1 to N / 4 do {
//       Q = fpminimax(asin(sqrt(i / N + x)) / sqrt(i / N + x), Deg,
//                     [| 128... | ], [ -1 / (2 * N), 1 / (2 * N) ]);
//       abs_err = dirtyinfnorm(asin(sqrt(i / N + x)) - sqrt(i / N + x) * Q,
//                              [ -1 / (2 * N), 1 / (2 * N) ]);
//       rel_err = dirtyinfnorm(asin(sqrt(i / N + x)) / sqrt(i / N + x) - Q,
//                              [ -1 / (2 * N), 1 / (2 * N) ]);
//       if (abs_err > abs_error) then abs_error = abs_err;
//       if (rel_err > rel_error) then rel_error = rel_err;
//       write("{");
//       for j from 0 to Deg do PRINTF128(coeff(Q, j));
//       print("},");
//     };
//     print("Absolute Errors:", abs_error);
//     print("Relative Errors:", rel_error);
//   };
// > ASIN_APPROX(64, 15);
// ...
// Absolute Errors: 0x1.0b3...p-129
// Relative Errors: 0x1.1db...p-128
//
// For k = 0, we use Taylor polynomial of asin(x)/x around x = 0.
//   asin(x)/x ~ 1 + x^2/6 + (3 x^4)/40 + (5 x^6)/112 + (35 x^8)/1152 +
//               + (63 x^10)/2816 + (231 x^12)/13312 + (143 x^14)/10240 +
//               + (6435 x^16)/557056 + (12155 x^18)/1245184 +
//               + (46189 x^20)/5505024 + (88179 x^22)/12058624 +
//               + (676039 x^24)/104857600 + (1300075 x^26)/226492416 +
//               + (5014575 x^28)/973078528 + (9694845 x^30)/2080374784.

constexpr Float128 ASIN_COEFFS_F128[17][16] = {
    {
        {Sign::POS, -127, 0x80000000'00000000'00000000'00000000_u128},
        {Sign::POS, -130, 0xaaaaaaaa'aaaaaaaa'aaaaaaaa'aaaaaaab_u128},
        {Sign::POS, -131, 0x99999999'99999999'99999999'9999999a_u128},
        {Sign::POS, -132, 0xb6db6db6'db6db6db'6db6db6d'b6db6db7_u128},
        {Sign::POS, -133, 0xf8e38e38'e38e38e3'8e38e38e'38e38e39_u128},
        {Sign::POS, -133, 0xb745d174'5d1745d1'745d1745'd1745d17_u128},
        {Sign::POS, -133, 0x8e276276'27627627'62762762'76276276_u128},
        {Sign::POS, -134, 0xe4cccccc'cccccccc'cccccccc'cccccccd_u128},
        {Sign::POS, -134, 0xbd43c3c3'c3c3c3c3'c3c3c3c3'c3c3c3c4_u128},
        {Sign::POS, -134, 0x9fef286b'ca1af286'bca1af28'6bca1af3_u128},
        {Sign::POS, -134, 0x89779e79'e79e79e7'9e79e79e'79e79e7a_u128},
        {Sign::POS, -135, 0xef9de9bd'37a6f4de'9bd37a6f'4de9bd38_u128},
        {Sign::POS, -135, 0xd3431eb8'51eb851e'b851eb85'1eb851ec_u128},
        {Sign::POS, -135, 0xbc16ed09'7b425ed0'97b425ed'097b425f_u128},
        {Sign::POS, -135, 0xa8dd1846'9ee58469'ee58469e'e58469ee_u128},
        {Sign::POS, -135, 0x98b41def'7bdef7bd'ef7bdef7'bdef7bdf_u128},
    },
    {
        {Sign::POS, -127, 0x8055f060'94f0f05f'3ac3b927'50a701d9_u128},
        {Sign::POS, -130, 0xad19c2ea'e3dd2429'8d04f71d'b965ee1b_u128},
        {Sign::POS, -131, 0x9dfa882b'7b31af17'f9f19d33'0c45d24b_u128},
        {Sign::POS, -132, 0xbedd3b58'c9e605ef'1404e1f0'4ba57940_u128},
        {Sign::POS, -132, 0x83df2581'cb4fea82'b406f201'2fde6d5c_u128},
        {Sign::POS, -133, 0xc534fe61'9b82dd16'ed5d8a43'f7710526_u128},
        {Sign::POS, -133, 0x9b56fa62'88295ddf'ce8425fe'a04d733e_u128},
        {Sign::POS, -134, 0xfdeddb19'4a030da7'27158080'd24caf46_u128},
        {Sign::POS, -134, 0xd55827db'ff416ea8'042c4d8c'07cddeeb_u128},
        {Sign::POS, -134, 0xb71d73a9'f2ba0688'5eaeeae9'413a0f5f_u128},
        {Sign::POS, -134, 0x9fde87e2'ace91274'38f82666'd619c1ba_u128},
        {Sign::POS, -134, 0x8d876557'5e4626a1'1b621336'93587847_u128},
        {Sign::POS, -135, 0xfd801840'c8710595'6880fe13'a9657f8f_u128},
        {Sign::POS, -135, 0xe54245a9'4c8c2ebb'30488494'64b0e34d_u128},
        {Sign::POS, -135, 0xd11eb46f'4095a661'8890d123'15c96482_u128},
        {Sign::POS, -135, 0xc01a4201'467fbc0b'960618d5'ec2adaa8_u128},
    },
    {
        {Sign::POS, -127, 0x80ad1cbe'7878de11'4293301c'11ce9d49_u128},
        {Sign::POS, -130, 0xaf9ac0df'3d845544'0fe5e31b'9051d03e_u128},
        {Sign::POS, -131, 0xa28ceef8'd7297e05'f94773ad'f4a695c6_u128},
        {Sign::POS, -132, 0xc75a5b77'58b4b11d'396c68ad'6733022b_u128},
        {Sign::POS, -132, 0x8bde42a1'084a6674'50c5bceb'005d4b62_u128},
        {Sign::POS, -133, 0xd471cdae'e2f35a96'bd4bc513'e0ccdf2c_u128},
        {Sign::POS, -133, 0xa9fc6fd5'd204a4e3'e609940c'6b991b67_u128},
        {Sign::POS, -133, 0x8d242d97'ba12b492'e25c7e7c'0c3fcf60_u128},
        {Sign::POS, -134, 0xf0f1ba74'b149afc3'2f0bbab5'a20c6199_u128},
        {Sign::POS, -134, 0xd21b42fb'd8e9098d'19612692'9a043332_u128},
        {Sign::POS, -134, 0xba5e5492'7896a3e7'193a74d5'78631587_u128},
        {Sign::POS, -134, 0xa7a17ae7'fc707f45'910e7a5d'c95251f4_u128},
        {Sign::POS, -134, 0x98889a6a'b0370464'50c950d3'61d79ed7_u128},
        {Sign::POS, -134, 0x8c29330e'4318fd29'25c5b528'84e39e7c_u128},
        {Sign::POS, -134, 0x81e7bf48'b25bc7c0'b9204a4f'd4f5fa8b_u128},
        {Sign::POS, -135, 0xf2801b09'11bf0768'773996dd'5224d852_u128},
    },
    {
        {Sign::POS, -127, 0x81058e3e'f82ba622'ab81cd63'e1a91d57_u128},
        {Sign::POS, -130, 0xb22e7055'c80dd354'8a2f2e8e'860d3f33_u128},
        {Sign::POS, -131, 0xa753ce1a'7e3d1f57'247b37e6'03f93624_u128},
        {Sign::POS, -132, 0xd05c5604'8eca8d18'dcdd76b7'f4b1f185_u128},
        {Sign::POS, -132, 0x947cdd5e'f1d64df0'84f78df1'e2ecb854_u128},
        {Sign::POS, -133, 0xe5218370'2ebbf6e8'3727a755'57843b93_u128},
        {Sign::POS, -133, 0xba482553'383b92eb'186f78f1'8c35d6af_u128},
        {Sign::POS, -133, 0x9d2b034a'7266c6a1'54b78a98'1a547429_u128},
        {Sign::POS, -133, 0x8852f723'feea6046'e125f5a9'64e168e6_u128},
        {Sign::POS, -134, 0xf19c9891'6c896c99'732052fe'5c54e992_u128},
        {Sign::POS, -134, 0xd9cc81a5'c5ddf0f0'd651011e'a8ecd936_u128},
        {Sign::POS, -134, 0xc7173169'dcb6095f'a6160847'b595aaff_u128},
        {Sign::POS, -134, 0xb81cd3f6'4a422ebe'07aeb734'e4dcf3a1_u128},
        {Sign::POS, -134, 0xabf01b1c'd15932aa'698d4382'512318a9_u128},
        {Sign::POS, -134, 0xa1f1cf1b'd889a1ac'7120ca2f'bbbc1745_u128},
        {Sign::POS, -134, 0x99a1b838'e38fbf11'429a4350'76b7d191_u128},
    },
    {
        {Sign::POS, -127, 0x815f4e70'5c3e68f2'e84ed170'78211dfd_u128},
        {Sign::POS, -130, 0xb4d5a992'de1ac4da'16fe6024'3a6cc371_u128},
        {Sign::POS, -131, 0xac526184'bd558c65'66642dce'edc4b04a_u128},
        {Sign::POS, -132, 0xd9ed9b03'46ec0bab'429ea221'4774bbc1_u128},
        {Sign::POS, -132, 0x9dca410c'1efaeb74'87956685'dd5fe848_u128},
        {Sign::POS, -133, 0xf76e411b'a926fc02'7f942265'9c39a882_u128},
        {Sign::POS, -133, 0xcc71b004'eeb60c0f'1d387f76'44b46bf8_u128},
        {Sign::POS, -133, 0xaf527a40'6f1084fb'5019904e'd12d384d_u128},
        {Sign::POS, -133, 0x9a9304b0'd8a9de19'e1803691'269be22c_u128},
        {Sign::POS, -133, 0x8b3d37c0'dbde09ef'342ddf4f'e80dd3fb_u128},
        {Sign::POS, -134, 0xff2e9111'3a961c78'92297bab'cc257804_u128},
        {Sign::POS, -134, 0xed1fb643'f2ca31c1'b0a1553a'e077285a_u128},
        {Sign::POS, -134, 0xdeeb0f5e'81ad5e30'78d79ae3'83be1c18_u128},
        {Sign::POS, -134, 0xd3a13ba6'8ce9abfc'a66eb1fd'c0c760fd_u128},
        {Sign::POS, -134, 0xcaa8c381'd44bb44f'0ab25126'9a5fae10_u128},
        {Sign::POS, -134, 0xc36fb2c4'244401cf'10dd8a39'78ccbf7f_u128},
    },
    {
        {Sign::POS, -127, 0x81ba6750'6064f4dd'08015b7c'713688f0_u128},
        {Sign::POS, -130, 0xb791524b'd975fdd1'584037b7'103b42ca_u128},
        {Sign::POS, -131, 0xb18c26c5'3ced9856'db5bc672'cc95a64f_u128},
        {Sign::POS, -132, 0xe4199ce5'd25be89b'4a0ad208'da77022d_u128},
        {Sign::POS, -132, 0xa7d77999'0f80e3e9'7e97e9d1'0e337550_u128},
        {Sign::POS, -132, 0x85c3e039'8959c95b'e6e1e87f'7e6636b1_u128},
        {Sign::POS, -133, 0xe0b90ecd'95f7e6eb'a675bae0'628bd214_u128},
        {Sign::POS, -133, 0xc3edb6b4'ed0a684c'c7a3ee4d'f1dcd3f9_u128},
        {Sign::POS, -133, 0xafa274d2'e66e1f61'9e8ab3c7'7221214e_u128},
        {Sign::POS, -133, 0xa0dd903d'e110b71a'8a1fc9df'cc080308_u128},
        {Sign::POS, -133, 0x95e2f38c'60441961'72b90625'e3a37573_u128},
        {Sign::POS, -133, 0x8d9fe38f'2c705139'029f857c'9f628b2b_u128},
        {Sign::POS, -133, 0x8762410a'4967a974'6b609e83'7c025a39_u128},
        {Sign::POS, -133, 0x82b220be'd9ec0e5a'9ce9af7c'c65c94b9_u128},
        {Sign::POS, -134, 0xfe866073'2312c056'4265d82a'3afea10c_u128},
        {Sign::POS, -134, 0xf99b667c'5f8ef6a6'11fafa4d'5c76ebb3_u128},
    },
    {
        {Sign::POS, -127, 0x8216e353'2ffdf638'15d72316'a2f327f2_u128},
        {Sign::POS, -130, 0xba625eba'097ce944'7024c0a3'c873729b_u128},
        {Sign::POS, -131, 0xb704e369'5b95ce44'cde30106'90e92cc3_u128},
        {Sign::POS, -132, 0xeeecee6d'7298b8a3'075da5d7'456bdcde_u128},
        {Sign::POS, -132, 0xb2b78fb1'fcfdc273'1d1ac11c'e29c16f1_u128},
        {Sign::POS, -132, 0x90d21722'148fdaf5'0d566a01'0bb8784b_u128},
        {Sign::POS, -133, 0xf7681c54'9771ebb6'17686858'eb5e1caf_u128},
        {Sign::POS, -133, 0xdb5e45c0'52ec0c1c'ff28765e'd4c44bfb_u128},
        {Sign::POS, -133, 0xc7ff0dd7'a34ee29b'7cb689af'fe887bf5_u128},
        {Sign::POS, -133, 0xba4e6f37'a98a3e3f'f1175427'20f45c82_u128},
        {Sign::POS, -133, 0xb08f6e11'688e4174'b3d48abe'c0a6d5cd_u128},
        {Sign::POS, -133, 0xa9af6a33'14aabe45'26da1218'05bbb52e_u128},
        {Sign::POS, -133, 0xa4fd22fa'1b4f0d7f'1456af96'cbd0cde6_u128},
        {Sign::POS, -133, 0xa20229b4'7e9c2e39'22c49987'66a05c5a_u128},
        {Sign::POS, -133, 0xa0775ca8'4409c735'351d01f1'34467927_u128},
        {Sign::POS, -133, 0xa010d2d9'08428a53'53603f20'66c8b8ba_u128},
    },
    {
        {Sign::POS, -127, 0x8274cd6a'f25e642d'0b1a02fb'03f53f3e_u128},
        {Sign::POS, -130, 0xbd49d2c8'b9005b2a'ee795b17'92181a48_u128},
        {Sign::POS, -131, 0xbcc0ac23'98e00fd7'c40811f5'486aca6a_u128},
        {Sign::POS, -132, 0xfa756493'b381b917'6cdea268'e44dd2fd_u128},
        {Sign::POS, -132, 0xbe7fce1e'462b43c6'0537d6f7'138c87ac_u128},
        {Sign::POS, -132, 0x9d00958b'edc83095'b4cc907c'a92c30f1_u128},
        {Sign::POS, -132, 0x886a2440'ed93d825'333c19c2'6de36d73_u128},
        {Sign::POS, -133, 0xf616ebc0'4f576462'd9312544'e8fbe0fd_u128},
        {Sign::POS, -133, 0xe43f4c9d'ebb5d685'00903a00'7bd6ad39_u128},
        {Sign::POS, -133, 0xd8516eab'32337672'569b4e19'a44e795c_u128},
        {Sign::POS, -133, 0xd091fa04'954666ee'cc4da283'82e977c0_u128},
        {Sign::POS, -133, 0xcbf13442'c4c0f859'0449c2c4'2fc046fe_u128},
        {Sign::POS, -133, 0xc9c1d1b4'dea4c76c'd101e562'dc3af77f_u128},
        {Sign::POS, -133, 0xc9924d2a'b8ec37d9'80af1780'0fb63e4e_u128},
        {Sign::POS, -133, 0xcb24b252'1ff37e4a'41f35260'2b9ace95_u128},
        {Sign::POS, -133, 0xce2d87ac'194a6304'1658ed0e'4cdb8161_u128},
    },
    {
        {Sign::POS, -127, 0x82d4310f'f58b570d'266275fc'1d085c87_u128},
        {Sign::POS, -130, 0xc048c361'72bee7b0'8d2ca7e5'afe4f335_u128},
        {Sign::POS, -131, 0xc2c3ecca'216e290e'b99c5c53'5d48595a_u128},
        {Sign::POS, -131, 0x83611e8f'3adf2217'be3c342a'dfb1c562_u128},
        {Sign::POS, -132, 0xcb481202'8b0ba9aa'e586f73d'faea68e4_u128},
        {Sign::POS, -132, 0xaa727c9a'4caba65d'c8dc13ef'8bed52e4_u128},
        {Sign::POS, -132, 0x96b05462'efac126e'db6871d0'0be1eff9_u128},
        {Sign::POS, -132, 0x8a4f8752'9b3c9232'63eb1596'a2c83eb4_u128},
        {Sign::POS, -132, 0x828be6f4'1b14e6e6'8efc1012'2afe425a_u128},
        {Sign::POS, -133, 0xfbd2f055'9d699ea9'b572008e'1fb08088_u128},
        {Sign::POS, -133, 0xf71b3c70'dc4610e6'bc1e581c'817b88bd_u128},
        {Sign::POS, -133, 0xf5e8ebf6'3b0aef3f'97ba4c8f'e49b6f0a_u128},
        {Sign::POS, -133, 0xf7986238'1eb8bd7a'73577ed0'c05e4abf_u128},
        {Sign::POS, -133, 0xfbc3832a'a903cd65'a46ee523'f342c621_u128},
        {Sign::POS, -132, 0x811ea5f3'7409245e'1777fdd1'59b29f80_u128},
        {Sign::POS, -132, 0x85619588'b83c90ef'67740d6a'd2f372a8_u128},
    },
    {
        {Sign::POS, -127, 0x83351a49'8764656f'e1774024'a5e751a6_u128},
        {Sign::POS, -130, 0xc36057da'23d39c2b'336474e0'3a893914_u128},
        {Sign::POS, -131, 0xc913714c'a46cc0bf'3bdd68ba'53a309d4_u128},
        {Sign::POS, -131, 0x89f2254d'f1469d60'e1324bac'95db6742_u128},
        {Sign::POS, -132, 0xd92b27f6'38df6911'5842365c'c120cc63_u128},
        {Sign::POS, -132, 0xb94ff079'7848d391'486efffa'a6fbc37f_u128},
        {Sign::POS, -132, 0xa6c03919'862e8437'70f86a73'43da3a6e_u128},
        {Sign::POS, -132, 0x9bcb70c9'a378e97f'a59f25f3'ba202e33_u128},
        {Sign::POS, -132, 0x95b103b0'62aa9f64'ee2d6146'76020bc5_u128},
        {Sign::POS, -132, 0x92fa4a1c'7d7fd161'8f25aa4e'f65ca52f_u128},
        {Sign::POS, -132, 0x92d387a2'c5dd771d'4015ca29'e3eda1d9_u128},
        {Sign::POS, -132, 0x94c13c5c'997615c3'8a2f63c8'c314226f_u128},
        {Sign::POS, -132, 0x987b8c8f'5e9e7a5f'e8497909'd60d1194_u128},
        {Sign::POS, -132, 0x9ddb0978'da99e6ad'83d5eca2'9d079ef7_u128},
        {Sign::POS, -132, 0xa4d9aeee'4b512ed4'5ec95cd1'37ce3f22_u128},
        {Sign::POS, -132, 0xad602af3'1e14d681'8a267da2'57c030de_u128},
    },
    {
        {Sign::POS, -127, 0x839795b7'8f3005a4'689f57cc'd201f7dc_u128},
        {Sign::POS, -130, 0xc691cb89'3d75d3d5'a1892f2a'bf54ec45_u128},
        {Sign::POS, -131, 0xcfb46fc4'6d28c32c'9ae5ad3d'a7749dc8_u128},
        {Sign::POS, -131, 0x90f71352'c806c830'20edb8b2'7594386b_u128},
        {Sign::POS, -132, 0xe8473840'd511dc77'd63def5d'7f4de9c0_u128},
        {Sign::POS, -132, 0xc9c6eb30'aaf2b63d'ec20f671'8689534a_u128},
        {Sign::POS, -132, 0xb8dcfa84'eb6cab93'3023ddcc'b8f68a2f_u128},
        {Sign::POS, -132, 0xafde4094'c1a14390'9609a3ea'847225a9_u128},
        {Sign::POS, -132, 0xac1254e7'5852a836'b2aca5e5'0cfc484f_u128},
        {Sign::POS, -132, 0xac0d3ffa'd6171016'b1a12557'858663c1_u128},
        {Sign::POS, -132, 0xaf0877f9'0ca5c52f'fc54b5af'b5cbc350_u128},
        {Sign::POS, -132, 0xb498574f'af349a2b'f391ff83'b3570919_u128},
        {Sign::POS, -132, 0xbc87c7bb'34182440'280647cd'976affb0_u128},
        {Sign::POS, -132, 0xc6c5688f'58a42593'4569de36'0855c393_u128},
        {Sign::POS, -132, 0xd368b088'5bb9496a'dd7c92df'8798aaf7_u128},
        {Sign::POS, -132, 0xe272168a'c8dbe668'381542bf'fc24c266_u128},
    },
    {
        {Sign::POS, -127, 0x83fbb09c'fbb0ebf4'208c9037'70373f79_u128},
        {Sign::POS, -130, 0xc9de6f84'8e652b0b'3b2a2bb9'f7ce3de8_u128},
        {Sign::POS, -131, 0xd6ac93c7'6e215233'f184fdcc'e5872970_u128},
        {Sign::POS, -131, 0x987a35b9'87c02522'1927dee9'70fc6b18_u128},
        {Sign::POS, -132, 0xf8be450d'266409a9'2e534ffd'905f4424_u128},
        {Sign::POS, -132, 0xdc0c36d7'34415e3b'c5121c4d'4e28c17d_u128},
        {Sign::POS, -132, 0xcd551b98'81d982a8'1399d9ba'ddf55821_u128},
        {Sign::POS, -132, 0xc6f91e3f'428d6be3'646f3147'20445145_u128},
        {Sign::POS, -132, 0xc64f100c'85e1e8f1'6f501d1e'2155f872_u128},
        {Sign::POS, -132, 0xc9fe25ae'295f1f24'5924cf9a'036a31f2_u128},
        {Sign::POS, -132, 0xd157410e'fcc10fbb'fceb318a'b4990bd7_u128},
        {Sign::POS, -132, 0xdc0aeb56'ca679f92'3b3c44d8'99b1add7_u128},
        {Sign::POS, -132, 0xea05b383'bc339550'e5c5c34b'bfa416a1_u128},
        {Sign::POS, -132, 0xfb5e3897'5a5c8f62'280a90dc'9ebe9107_u128},
        {Sign::POS, -131, 0x88301d81'b38f225d'2226ab7e'df342d90_u128},
        {Sign::POS, -131, 0x949e3465'e4a8aef7'46311182'5fc3fde8_u128},
    },
    {
        {Sign::POS, -127, 0x846178eb'1c7260da'3e0aca9a'51e68d84_u128},
        {Sign::POS, -130, 0xcd47ac90'3c311c2b'98dd7493'4656d210_u128},
        {Sign::POS, -131, 0xde020b2d'abd5628c'b88634e5'73f312fc_u128},
        {Sign::POS, -131, 0xa086fafa'c220fb73'9939cae3'2d69683f_u128},
        {Sign::POS, -131, 0x855b5efa'f6963d73'e4664cb1'd43f03a9_u128},
        {Sign::POS, -132, 0xf05c9774'fe0de25c'ccf1c1df'd2ed9941_u128},
        {Sign::POS, -132, 0xe484a941'19639229'f06ae955'f8edc7d1_u128},
        {Sign::POS, -132, 0xe1a32bb2'52ca122c'bf2f0904'cfc476cb_u128},
        {Sign::POS, -132, 0xe528e091'7bb8a01a'9218ce3e'1e85af60_u128},
        {Sign::POS, -132, 0xeddd556a'faa2d46f'e91c61fa'adf12aec_u128},
        {Sign::POS, -132, 0xfb390fa3'15e9d55f'5683c0c4'c7719f81_u128},
        {Sign::POS, -131, 0x868e5fa4'15597c8f'7c42a262'8f2d6332_u128},
        {Sign::POS, -131, 0x91d79767'a3d037f9'cd84ead5'c0714310_u128},
        {Sign::POS, -131, 0x9fa6a035'915bc052'377a8abb'faf4e3c6_u128},
        {Sign::POS, -131, 0xb04edefd'6ac2a93e'ec33e6f6'3d53e7c2_u128},
        {Sign::POS, -131, 0xc416980d'dc5c186b'7bdcded6'97ea5844_u128},
    },
    {
        {Sign::POS, -127, 0x84c8fd4d'ffdf9fc6'bdd7ebca'88183d7b_u128},
        {Sign::POS, -130, 0xd0cf0544'11dbf845'cb6eeae5'bc980e2f_u128},
        {Sign::POS, -131, 0xe5bb9480'7ce0eaca'74300a46'8398e944_u128},
        {Sign::POS, -131, 0xa92a18f8'd611860b'5f2ef8c6'8e8ca002_u128},
        {Sign::POS, -131, 0x8f2e1684'17eb4e6c'1ec44b9b'e4b1c3e5_u128},
        {Sign::POS, -131, 0x837f1764'0ee8f416'8694b4a1'c647af0c_u128},
        {Sign::POS, -132, 0xfed7e2a9'05a5190e'b7d70a61'a24ad801_u128},
        {Sign::POS, -131, 0x803f29ff'dc6fd2bc'3c3c4b50'a9dc860c_u128},
        {Sign::POS, -131, 0x84c61e09'b8aa35e4'96239f9c'b1d00b3c_u128},
        {Sign::POS, -131, 0x8c7ed311'f77980d6'842ddf90'6a68a0bc_u128},
        {Sign::POS, -131, 0x9746077b'd397c2d1'038a4744'a76f5fb5_u128},
        {Sign::POS, -131, 0xa5341277'c4185ace'54f26328'322158e8_u128},
        {Sign::POS, -131, 0xb68d78f5'0972f6de'9189aa23'd3ecefc2_u128},
        {Sign::POS, -131, 0xcbbcefc2'15bade4e'f1d36947'c8b6e460_u128},
        {Sign::POS, -131, 0xe564a459'c851390d'd45a4748'f29f182b_u128},
        {Sign::POS, -130, 0x820ea28b'c89662c3'2a64ccdc'efb2b259_u128},
    },
    {
        {Sign::POS, -127, 0x85324d39'f30f9174'ac0d817e'9c744b0b_u128},
        {Sign::POS, -130, 0xd476186e'49c47f3a'a71f8886'7f9f21c4_u128},
        {Sign::POS, -131, 0xede08f54'a830e87b'07881700'65e57b6c_u128},
        {Sign::POS, -131, 0xb271b8eb'309963ee'89187c73'0b92f7d5_u128},
        {Sign::POS, -131, 0x99f0011d'95d3a6dd'282bd00a'db808151_u128},
        {Sign::POS, -131, 0x9021134e'02b479e7'3aabf9bb'b7ab6cf3_u128},
        {Sign::POS, -131, 0x8e673bf2'f11db54a'909c4c72'6389499f_u128},
        {Sign::POS, -131, 0x9226a371'88dd55f7'bfe21777'4a42a7ae_u128},
        {Sign::POS, -131, 0x9a4d78fc'9df79d9a'44609c02'a625808a_u128},
        {Sign::POS, -131, 0xa68335fb'41d2d91c'e7bbd2a3'31a1d17b_u128},
        {Sign::POS, -131, 0xb6d89c39'28d0cb26'809d4df6'e55cba1a_u128},
        {Sign::POS, -131, 0xcba71468'9177fc2d'7f23df2f'37226488_u128},
        {Sign::POS, -131, 0xe5846de8'44833ae9'34416c87'0315eb9e_u128},
        {Sign::POS, -130, 0x82a07032'64e6226b'200d94a1'66fc7951_u128},
        {Sign::POS, -130, 0x9602695c'b6fa8886'68ca0cba'b59ea683_u128},
        {Sign::POS, -130, 0xad7d185a'ab3d14dd'd908a7b1'c57352bb_u128},
    },
    {
        {Sign::POS, -127, 0x859d78fa'4405d8fa'287dbc69'95d0975e_u128},
        {Sign::POS, -130, 0xd83ea3bc'131d6baa'67c51d88'4c4dae01_u128},
        {Sign::POS, -131, 0xf6790edb'df07342b'aad85870'167af128_u128},
        {Sign::POS, -131, 0xbc6daa33'12be0f85'bc7fa753'52b10a83_u128},
        {Sign::POS, -131, 0xa5bd41bc'9c986b13'1af2542e'92aacb59_u128},
        {Sign::POS, -131, 0x9e4358bc'24e04364'b4539b76'e444b790_u128},
        {Sign::POS, -131, 0x9f7fc21b'dca1f2b5'f3f6d44b'c5a37626_u128},
        {Sign::POS, -131, 0xa6fd793c'0b9c44c1'30a518cc'66b5e511_u128},
        {Sign::POS, -131, 0xb3dccfac'cd1592b3'bcd6b7c0'9749993d_u128},
        {Sign::POS, -131, 0xc6056c3a'4a5f329a'48f1429d'27f930fc_u128},
        {Sign::POS, -131, 0xddd9e529'858a4502'6e7f3d1c'1e7dcb89_u128},
        {Sign::POS, -131, 0xfc1bccee'dc8d2567'1721c468'6f7f53ec_u128},
        {Sign::POS, -130, 0x90f2bb21'5cdbe7e2'f9ef8e12'059cc66a_u128},
        {Sign::POS, -130, 0xa857d5df'5b4da940'15ce4e95'7201fc79_u128},
        {Sign::POS, -130, 0xc54119c0'10c02bf4'd87ece17'1ef85c5f_u128},
        {Sign::POS, -130, 0xe8c50ebc'880356de'2c1f4c42'9ee9748f_u128},
    },
    {
        {Sign::POS, -127, 0x860a91c1'6b9b2c23'2dd99707'ab3d688b_u128},
        {Sign::POS, -130, 0xdc2a86b1'5fdb645d'ea2781dd'25555f49_u128},
        {Sign::POS, -131, 0xff8def07'd1e514d7'b2e8ebb6'5c3afe5e_u128},
        {Sign::POS, -131, 0xc72f9d5b'4fb559e3'20db92e3'a5ae3f73_u128},
        {Sign::POS, -131, 0xb2b5f45b'1d26f4dd'0b210309'fb68914f_u128},
        {Sign::POS, -131, 0xae1cbaae'c7b55465'4da858f5'47e62a37_u128},
        {Sign::POS, -131, 0xb30f3998'10202a0d'a52ec085'a7d63289_u128},
        {Sign::POS, -131, 0xbf51f27f'b7aff89d'dc24e2aa'208d2054_u128},
        {Sign::POS, -131, 0xd250735e'87d0b527'6f99bcc9'bd6fc717_u128},
        {Sign::POS, -131, 0xec543ec2'bddb2efb'36d9ce81'a7c84336_u128},
        {Sign::POS, -130, 0x871f73e3'298ef45c'eed83998'2bc731b9_u128},
        {Sign::POS, -130, 0x9cbb5447'af8574f1'21fa4cda'93d82b7e_u128},
        {Sign::POS, -130, 0xb7f5a6c0'430a347f'11b22cde'91de0885_u128},
        {Sign::POS, -130, 0xda153cc4'14abdb96'840df7c2'3299fec0_u128},
        {Sign::POS, -129, 0x826c129b'3e4a2612'b2cd11f1'4d2ba60c_u128},
        {Sign::POS, -129, 0x9d19c289'fc0e8aa4'f351418b'b760ce90_u128},
    },
};

constexpr Float128 PI_OVER_TWO_F128 = {
    Sign::POS, -127, 0xc90fdaa2'2168c234'c4c6628b'80dc1cd1_u128};

constexpr Float128 PI_F128 = {Sign::POS, -126,
                              0xc90fdaa2'2168c234'c4c6628b'80dc1cd1_u128};

LIBC_INLINE Float128 asin_eval(const Float128 &u, unsigned idx) {
  return fputil::polyeval(u, ASIN_COEFFS_F128[idx][0], ASIN_COEFFS_F128[idx][1],
                          ASIN_COEFFS_F128[idx][2], ASIN_COEFFS_F128[idx][3],
                          ASIN_COEFFS_F128[idx][4], ASIN_COEFFS_F128[idx][5],
                          ASIN_COEFFS_F128[idx][6], ASIN_COEFFS_F128[idx][7],
                          ASIN_COEFFS_F128[idx][8], ASIN_COEFFS_F128[idx][9],
                          ASIN_COEFFS_F128[idx][10], ASIN_COEFFS_F128[idx][11],
                          ASIN_COEFFS_F128[idx][12], ASIN_COEFFS_F128[idx][13],
                          ASIN_COEFFS_F128[idx][14], ASIN_COEFFS_F128[idx][15]);
}

#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

} // anonymous namespace

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_GENERIC_ASIN_UTILS_H
