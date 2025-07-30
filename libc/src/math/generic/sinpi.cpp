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
  if (x_abs > 0x4320000000000000) {
    if (x_abs < 0x4330000000000000) {
      if ((x_abs & 1) == 0)
        return 0.0;
      return (x_abs & 2) ? -1.0 : 1.0;
    }

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
  DoubleDouble sin_k = SIN_K_PI_OVER_128[k_int & 255];
  DoubleDouble cos_k = SIN_K_PI_OVER_128[(k_int + 64) & 255];

  double cosm1_y = cos_y.hi - 1.0;
  DoubleDouble sin_y_cos_k = fputil::quick_mult(sin_y, cos_k);

  DoubleDouble cosm1_yy;
  cosm1_yy.hi = cosm1_y;
  cosm1_yy.lo = 0.0;

  DoubleDouble cos_y_sin_k = fputil::quick_mult(cos_y, sin_k);
  DoubleDouble rr = fputil::exact_add<false>(sin_y_cos_k.hi, cos_y_sin_k.hi);

  rr.lo += sin_y_cos_k.lo + cos_y_sin_k.lo;

  return rr.hi + rr.lo;
}
} // namespace LIBC_NAMESPACE_DECL
