#include "src/math/sinpi.h"
#include "range_reduction_double_nofma.h"
#include "sincos_eval.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/math/pow.h"
// #include "src/__support/FPUtil/multiply_add.h"
// #include "src/math/generic/range_reduction_double_common.h"
#include <iostream>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, sinpi, (double x)) {
  // Given x * pi = y - (k * (pi/128))
  // find y and k such that
  // y = x * pi - kpi/128
  // k = round(x, 128)

  using FPBits = typename fputil::FPBits<double>;
  using DoubleDouble = fputil::DoubleDouble;

  FPBits xbits(x);

  double k = fputil::nearest_integer(x * 128);
  int k_int = static_cast<int>(k);

  std::cout << "k" << k << std::endl;

  double yk = x - k / 128;

  // using Veltkamp splitting, we can use sollya to split pi:
  //    > x_h = round(pi, D, RN);
  //    > x_l = round(pi-x_h, D, RN);
  DoubleDouble yy = fputil::exact_mult(
      yk, 3.141592653589793115997963468544185161590576171875);
  yy.lo = fputil::multiply_add(
      yk, 1.2246467991473532071737640294583966046256921246776e-16, yy.lo);

  uint64_t abs_u = xbits.uintval();

  uint64_t x_abs = abs_u & 0xFFFFFFFFFFFFFFFF;

  if (LIBC_UNLIKELY(x_abs == 0U))
    return x;
  // When |x| > 2^51, x is an Integer or Nan/Inf
  if (x_abs >= 0x4320000000000000) {
    if (x_abs < 0x4330000000000000)
      return (x_abs & 0x1) ? -1.0 : 1.0;
    // |x| >= 2^52
    if (x_abs >= 0x4330000000000000) {
      if (xbits.is_nan())
        return x;
      if (xbits.is_inf()) {
        fputil::set_errno_if_required(EDOM);
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
    }
    return FPBits::zero(xbits.sign()).get_val();
  }

  DoubleDouble sin_y, cos_y;

  [[maybe_unused]] double err = generic::sincos_eval(yy, sin_y, cos_y);
  DoubleDouble sin_k = SIN_K_PI_OVER_128[k_int & 255];
  DoubleDouble cos_k = SIN_K_PI_OVER_128[(k_int + 64) & 255];

  std::cout << "sin_k: " << sin_k.hi << std::endl;
  std::cout << "sin_klo: " << sin_k.lo << std::endl;
  std::cout << "sin_y: " << sin_y.hi << std::endl;
  std::cout << "cos_y: " << cos_y.hi << std::endl;
  std::cout << "sin_y.lo: " << sin_y.lo << std::endl;
  std::cout << "cos_y.o: " << cos_y.lo << std::endl;

  double cosm1_y = cos_y.hi - 1.0;
  DoubleDouble sin_y_cos_k = fputil::quick_mult(sin_y, cos_k);

  std::cout << "cosm1" << cosm1_y << std::endl;
  DoubleDouble cosm1_yy;
  cosm1_yy.hi = cosm1_y;
  cosm1_yy.lo = 0.0;

  DoubleDouble cos_y_sin_k = fputil::quick_mult(cos_y, sin_k);
  DoubleDouble rr = fputil::exact_add<false>(sin_y_cos_k.hi, cos_y_sin_k.hi);

  std::cout << "r.hi:" << rr.hi << std::endl;
  std::cout << "r.lo" << rr.lo << std::endl;

  rr.lo += sin_y_cos_k.lo + cos_y_sin_k.lo;

  std::cout << "rrlo2: " << rr.lo << std::endl;
  std::cout << "cos_y_sin_k:" << cos_y_sin_k.hi << std::endl;
  std::cout << "siny*cosk.lo:" << sin_y_cos_k.lo << std::endl;
  std::cout << "rrhi + rrlo + sink.hi " << rr.hi + rr.lo + sin_k.hi + sin_k.lo
            << std::endl;
  std::cout << "rrhi + rrlo " << rr.hi + rr.lo << std::endl;

  return rr.hi + rr.lo;
}
} // namespace LIBC_NAMESPACE_DECL
