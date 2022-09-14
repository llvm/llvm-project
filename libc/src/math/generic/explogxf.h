//===-- Single-precision general exp/log functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_EXPLOGXF_H
#define LLVM_LIBC_SRC_MATH_GENERIC_EXPLOGXF_H

#include "math_utils.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/common.h"
#include <src/__support/FPUtil/NearestIntegerOperations.h>

#include <errno.h>

namespace __llvm_libc {

static constexpr int EXP_bits_p = 5;
static constexpr int EXP_num_p = 1 << EXP_bits_p;
constexpr double mlp = EXP_num_p;
constexpr double mmld = -1.0 / mlp;

// Wolfram alpha: N[Table[2^x-1,{x,-16/32,15/32,1/32}],27]
// printf("%.13a,\n", d[i]);
extern const double EXP_2_POW[EXP_num_p];

constexpr int LOG_P1_BITS = 6;
constexpr int LOG_P1_SIZE = 1 << LOG_P1_BITS;

// N[Table[Log[2, 1 + x], {x, 0/64, 63/64, 1/64}], 40]
extern const double LOG_P1_LOG2[LOG_P1_SIZE];

// N[Table[1/(1 + x), {x, 0/64, 63/64, 1/64}], 40]
extern const double LOG_P1_1_OVER[LOG_P1_SIZE];

// Taylor series expansion for Log[2, 1 + x] splitted to EVEN AND ODD numbers
// K_LOG2_ODD starts from x^3
extern const double K_LOG2_ODD[4];
extern const double K_LOG2_EVEN[4];

// The algorithm represents exp(x) as
//   exp(x) = 2^(ln(2) * i) * 2^(ln(2) * j / NUM_P )) * exp(dx)
// where i integer value, j integer in range [-NUM_P/2, NUM_P/2).
// 2^(ln(2) * j / NUM_P )) is a table values: 1.0 + EXP_M
// exp(dx) calculates by taylor expansion.

// Inversion of ln(2). Multiplication by EXP_num_p due to sampling by 1 /
// EXP_num_p Precise value of the constant is not needed.
static constexpr double LN2_INV = 0x1.71547652b82fep+0 * EXP_num_p;

// LN2_HIGH + LN2_LOW = ln(2) with precision higher than double(ln(2))
// Minus sign is to use FMA directly.
static constexpr double LN2_HIGH = -0x1.62e42fefa0000p-1 / EXP_num_p;
static constexpr double LN2_LOW = -0x1.cf79abc9e3b3ap-40 / EXP_num_p;

struct exe_eval_result_t {
  // exp(x) = 2^MULT_POWER2 * mult_exp * (r + 1.0)
  // where
  //   MULT_POWER2 template parameter;
  //   mult_exp = 2^e;
  //   r in range [~-0.3, ~0.41]
  double mult_exp;
  double r;
};

// The function correctly calculates exp value with at least float precision
// in range not narrow than [-log(2^-150), 90]
template <int MULT_POWER2 = 0>
inline static exe_eval_result_t exp_eval(double x) {
  double ps_dbl = fputil::nearest_integer(LN2_INV * x);
  // Negative sign due to multiply_add optimization
  double mult_e1, ml;
  {
    int ps =
        static_cast<int>(ps_dbl) + (1 << (EXP_bits_p - 1)) +
        ((fputil::FPBits<double>::EXPONENT_BIAS + MULT_POWER2) << EXP_bits_p);
    int table_index = ps & (EXP_num_p - 1);
    fputil::FPBits<double> bs;
    bs.set_unbiased_exponent(ps >> EXP_bits_p);
    ml = EXP_2_POW[table_index];
    mult_e1 = bs.get_val();
  }
  double dx = fputil::multiply_add(ps_dbl, LN2_LOW,
                                   fputil::multiply_add(ps_dbl, LN2_HIGH, x));

  // Taylor series coefficients
  double pe = dx * fputil::polyeval(dx, 1.0, 0x1.0p-1, 0x1.5555555555555p-3,
                                    0x1.5555555555555p-5, 0x1.1111111111111p-7,
                                    0x1.6c16c16c16c17p-10);

  double r = fputil::multiply_add(ml, pe, pe) + ml;
  return {mult_e1, r};
}

// x should be positive, normal finite value
inline static double log2_eval(double x) {
  using FPB = fputil::FPBits<double>;
  FPB bs(x);

  double result = 0;
  result += bs.get_exponent();

  int p1 =
      (bs.get_mantissa() >> (FPB::FloatProp::MANTISSA_WIDTH - LOG_P1_BITS)) &
      (LOG_P1_SIZE - 1);

  bs.bits &= FPB::FloatProp::MANTISSA_MASK >> LOG_P1_BITS;
  bs.set_unbiased_exponent(FPB::FloatProp::EXPONENT_BIAS);
  double dx = (bs.get_val() - 1.0) * LOG_P1_1_OVER[p1];

  // Taylor series for log(2,1+x)
  double c1 = fputil::multiply_add(dx, K_LOG2_ODD[0], K_LOG2_EVEN[0]);
  double c2 = fputil::multiply_add(dx, K_LOG2_ODD[1], K_LOG2_EVEN[1]);
  double c3 = fputil::multiply_add(dx, K_LOG2_ODD[2], K_LOG2_EVEN[2]);
  double c4 = fputil::multiply_add(dx, K_LOG2_ODD[3], K_LOG2_EVEN[3]);

  // c0 = dx * (1.0 / ln(2)) + LOG_P1_LOG2[p1]
  double c0 = fputil::multiply_add(dx, 0x1.71547652b82fep+0, LOG_P1_LOG2[p1]);
  result += __llvm_libc::fputil::polyeval(dx * dx, c0, c1, c2, c3, c4);
  return result;
}

// x should be positive, normal finite value
inline static double log_eval(double x) {
  // ln(x) = log[2,x] * ln(2)
  return log2_eval(x) * 0x1.62e42fefa39efp-1;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GENERIC_EXPLOGXF_H
