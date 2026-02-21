//===-- Implementation header for sqrti -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_SQRTI_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_SQRTI_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T> sqrti(T n) {
  if (n == 0)
    return 0;

  // Upper bound shift so that (1 << shift) >= sqrt(n).
  // log2_n approx = (bits - 1) - clz(n)
  // shift = (log2_n / 2) + 1, which is >= ceil(log2_n / 2)
  int bits = static_cast<int>(cpp::numeric_limits<T>::digits);
  int lz = cpp::countl_zero(n);
  int log2_n = bits - 1 - lz;
  int shift = (log2_n / 2) + 1;

  // 'x' is guaranteed to be >= sqrt(n).
  // This satisfies the condition for Newton-Raphson to converge monotonically.
  T x = T(1) << shift;

  // Newton-Raphson Iteration: x_{k+1} = (x_k + n / x_k) / 2
  // the sequence decreases to floor(sqrt(n)).
  while (true) {
    T next = (x + n / x) / 2;
    if (next >= x)
      return x;
    x = next;
  }
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_SQRTI_H
