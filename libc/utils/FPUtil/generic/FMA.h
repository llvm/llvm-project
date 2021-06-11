//===-- Common header for FMA implementations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_GENERIC_FMA_H
#define LLVM_LIBC_UTILS_FPUTIL_GENERIC_FMA_H

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {
namespace generic {

template <typename T>
static inline cpp::EnableIfType<cpp::IsSame<T, float>::Value, T> fma(T x, T y,
                                                                     T z) {
  // Product is exact.
  double prod = static_cast<double>(x) * static_cast<double>(y);
  double z_d = static_cast<double>(z);
  double sum = prod + z_d;
  fputil::FPBits<double> bit_prod(prod), bitz(z_d), bit_sum(sum);

  if (!(bit_sum.isInfOrNaN() || bit_sum.isZero())) {
    // Since the sum is computed in double precision, rounding might happen
    // (for instance, when bitz.exponent > bit_prod.exponent + 5, or
    // bit_prod.exponent > bitz.exponent + 40).  In that case, when we round
    // the sum back to float, double rounding error might occur.
    // A concrete example of this phenomenon is as follows:
    //   x = y = 1 + 2^(-12), z = 2^(-53)
    // The exact value of x*y + z is 1 + 2^(-11) + 2^(-24) + 2^(-53)
    // So when rounding to float, fmaf(x, y, z) = 1 + 2^(-11) + 2^(-23)
    // On the other hand, with the default rounding mode,
    //   double(x*y + z) = 1 + 2^(-11) + 2^(-24)
    // and casting again to float gives us:
    //   float(double(x*y + z)) = 1 + 2^(-11).
    //
    // In order to correct this possible double rounding error, first we use
    // Dekker's 2Sum algorithm to find t such that sum - t = prod + z exactly,
    // assuming the (default) rounding mode is round-to-the-nearest,
    // tie-to-even.  Moreover, t satisfies the condition that t < eps(sum),
    // i.e., t.exponent < sum.exponent - 52. So if t is not 0, meaning rounding
    // occurs when computing the sum, we just need to use t to adjust (any) last
    // bit of sum, so that the sticky bits used when rounding sum to float are
    // correct (when it matters).
    fputil::FPBits<double> t(
        (bit_prod.encoding.exponent >= bitz.encoding.exponent)
            ? ((double(bit_sum) - double(bit_prod)) - double(bitz))
            : ((double(bit_sum) - double(bitz)) - double(bit_prod)));

    // Update sticky bits if t != 0.0 and the least (52 - 23 - 1 = 28) bits are
    // zero.
    if (!t.isZero() && ((bit_sum.encoding.mantissa & 0xfff'ffffULL) == 0)) {
      if (bit_sum.encoding.sign != t.encoding.sign) {
        ++bit_sum.encoding.mantissa;
      } else if (bit_sum.encoding.mantissa) {
        --bit_sum.encoding.mantissa;
      }
    }
  }

  return static_cast<float>(static_cast<double>(bit_sum));
}

} // namespace generic
} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_GENERIC_FMA_H
