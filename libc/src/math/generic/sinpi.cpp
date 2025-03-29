#include "src/math/sinpi.h"
#include "sincos_eval.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/math/pow.h"
#include "range_reduction_double_nofma.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/math/generic/range_reduction_double_common.h"
#include <iostream>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, sinpi, (double x)) {

  // Range reduction:
  // Given x = (k + y) * 1/128
  // find k and y such that
  //   k = round(x * 128)
  //   y = x * 128 - k

  // x = (k + y) * 1/128 and
  // sin(x * pi) = sin((k +y)*pi/128)
  //             = sin(k * pi/128) * cos(y * pi/128) +
  //             = sin(y * pi/128) * cos(k* pi/128)

  using FPBits = typename fputil::FPBits<double>;
  using DoubleDouble = fputil::DoubleDouble;

  LargeRangeReduction range_reduction_large{};
  double k = fputil::nearest_integer(x * 128);
  FPBits kbits(k);
  FPBits xbits(x);
  uint64_t k_bits = kbits.uintval();

  double fff = 5.0;
  [[maybe_unused]] Float128 ggg = range_reduction_small_f128(fff);

  double y = (x * 128) - k;
  constexpr DoubleDouble PI_OVER_128_DD = {0x1.1a62633145c07p-60,
                                           0x1.921fb54442d18p-6};
  double pi_over_128 = PI_OVER_128_DD.hi;
  DoubleDouble yy = fputil::exact_mult(y, pi_over_128);

  uint64_t abs_u = xbits.uintval();

  uint64_t x_abs = abs_u & 0xFFFFFFFFFFFFFFFF;

  if (LIBC_UNLIKELY(x_abs == 0U))
    return x;

  if (x_abs >= 0x4330000000000000) {
    if (xbits.is_nan())
      return x;
    if (xbits.is_inf()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return FPBits::zero(xbits.sign()).get_val();
  }

  DoubleDouble sin_y, cos_y;

  [[maybe_unused]] double err = generic::sincos_eval(yy, sin_y, cos_y);
  DoubleDouble sin_k = SIN_K_PI_OVER_128[k_bits & 255];
  DoubleDouble cos_k = SIN_K_PI_OVER_128[(k_bits + 64) & 255];
  
  DoubleDouble sin_k_cos_y = fputil::quick_mult(cos_y, sin_k);
  DoubleDouble cos_k_sin_y = fputil::quick_mult(sin_y, cos_k);
  
  
  DoubleDouble rr = fputil::exact_add<false>(sin_k_cos_y.hi, cos_k_sin_y.hi);
  rr.lo += sin_k_cos_y.lo + cos_k_sin_y.lo;

  double rlp = rr.lo + err;
  double rlm = rr.lo - err;

  double r_upper = rr.hi + rlp; // (rr.lo + ERR);
  double r_lower = rr.hi + rlm; // (rr.lo - ERR);

  uint16_t x_e = xbits.get_biased_exponent();

  // Ziv's rounding test.
  if (LIBC_LIKELY(r_upper == r_lower))
    return r_upper;

  Float128 u_f128, sin_u, cos_u;
  if (LIBC_LIKELY(x_e < FPBits::EXP_BIAS + FAST_PASS_EXPONENT))
    u_f128 = range_reduction_small_f128(x);
  else
    u_f128 = range_reduction_large.accurate();

  generic::sincos_eval(u_f128, sin_u, cos_u);

  auto get_sin_k = [](unsigned kk) -> Float128 {
    unsigned idx = (kk & 64) ? 64 - (kk & 63) : (kk & 63);
    Float128 ans = SIN_K_PI_OVER_128_F128[idx];
    if (kk & 128)
      ans.sign = Sign::NEG;
    return ans;
  };

  unsigned k_r = range_reduction_large.fast(x, yy);
  std::cout << k_r << "k_r" << std::endl;
  // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
  Float128 sin_k_f128 = get_sin_k(k_r);
  Float128 cos_k_f128 = get_sin_k(k_r + 64);

  // sin(x) = sin((k * pi/128 + u)
  //        = sin(u) * cos(k*pi/128) + cos(u) * sin(k*pi/128)
  Float128 r = fputil::quick_add(fputil::quick_mul(sin_k_f128, cos_u),
                                 fputil::quick_mul(cos_k_f128, sin_u));

  // TODO: Add assertion if Ziv's accuracy tests fail in debug mode.
  // https://github.com/llvm/llvm-project/issues/96452.

  return static_cast<double>(r);
  
  //double cos_ulp = cos_k.lo + err;
  //double sin_ulp = sin_k.hi + err;
  /*
  double sin_yy = sin_y.hi;
  double cos_yy = cos_y.hi;
  double sin_kk = sin_k.hi;
  double cos_kk = cos_k.hi;
  */
  /*
  if (LIBC_UNLIKELY(sin_yy == 0 && sin_kk == 0))
    return FPBits::zero(xbits.sign()).get_val();
  */
  /*
  return static_cast<double>(fputil::multiply_add(
      sin_yy, cos_kk, fputil::multiply_add(cos_yy, sin_kk, 0.0)));
  */

}
} // namespace LIBC_NAMESPACE_DECL
