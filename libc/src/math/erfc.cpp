#include "src/math/erfc.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, erfc, (double x)) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);

  // 1. Handle Edge Cases exactly as required by standard libraries
  if (xbits.is_nan()) {
    return x;
  }
  if (xbits.is_inf()) {
    return xbits.is_pos() ? 0.0 : 2.0;
  }
  if (xbits.is_zero()) {
    return 1.0;
  }

  double abs_x = xbits.abs().get_val();

  // 2. Handle extreme limits where floating-point math breaks down
  if (abs_x > 27.0) {
    return x > 0.0 ? 0.0 : 2.0;
  }

  // 3. Polynomial Evaluation (No loops!)
  // We use a mathematical fractional approximation to avoid the Taylor Series.
  // t = 1 / (1 + p * x)
  constexpr double P = 0.3275911;
  double t = 1.0 / (1.0 + P * abs_x);

  // fputil::polyeval automatically turns these coefficients into highly
  // optimized machine code using Horner's Method.
  double poly = fputil::polyeval(t,
                                 0.0,          // t^0 (Starts at 0)
                                 0.254829592,  // t^1
                                 -0.284496736, // t^2
                                 1.421413741,  // t^3
                                 -1.453152027, // t^4
                                 1.061405429   // t^5
  );

  // 4. Calculate the final result directly to avoid catastrophic cancellation.
  // Note: We use __builtin_exp for this isolated example. In a full LLVM
  // environment, you would call their internal exp implementation.
  double result = poly * __builtin_exp(-abs_x * abs_x);

  // If x was negative, the result mirrors across 2.0
  return x > 0.0 ? result : 2.0 - result;
}

} // namespace LIBC_NAMESPACE_DECL
