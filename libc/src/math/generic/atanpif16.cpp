//===-- Half-precision atanpi function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanpif16.h"
#include "src/__support/math/atanpif16.h"

namespace LIBC_NAMESPACE_DECL {

// Using Python's SymPy library, we can obtain the polynomial approximation of
// arctan(x)/pi. The steps are as follows:
//  >>> from sympy import *
//  >>> import math
//  >>> x = symbols('x')
//  >>> print(series(atan(x)/math.pi, x, 0, 17))
//
// Output:
// 0.318309886183791*x - 0.106103295394597*x**3 + 0.0636619772367581*x**5 -
// 0.0454728408833987*x**7 + 0.0353677651315323*x**9 - 0.0289372623803446*x**11
// + 0.0244853758602916*x**13 - 0.0212206590789194*x**15 + O(x**17)
//
// We will assign this degree-15 Taylor polynomial as g(x). This polynomial
// approximation is accurate for arctan(x)/pi when |x| is in the range [0, 0.5].
//
//
// To compute arctan(x) for all real x, we divide the domain into the following
// cases:
//
// * Case 1: |x| <= 0.5
//      In this range, the direct polynomial approximation is used:
//      arctan(x)/pi = sign(x) * g(|x|)
//      or equivalently, arctan(x) = sign(x) * pi * g(|x|).
//
// * Case 2: 0.5 < |x| <= 1
//      We use the double-angle identity for the tangent function, specifically:
//        arctan(x) = 2 * arctan(x / (1 + sqrt(1 + x^2))).
//      Applying this, we have:
//        arctan(x)/pi = sign(x) * 2 * arctan(x')/pi,
//        where x' = |x| / (1 + sqrt(1 + x^2)).
//        Thus, arctan(x)/pi = sign(x) * 2 * g(x')
//
//      When |x| is in (0.5, 1], the value of x' will always fall within the
//      interval [0.207, 0.414], which is within the accurate range of g(x).
//
// * Case 3: |x| > 1
//      For values of |x| greater than 1, we use the reciprocal transformation
//      identity:
//        arctan(x) = pi/2 - arctan(1/x) for x > 0.
//      For any x (real number), this generalizes to:
//        arctan(x)/pi = sign(x) * (1/2 - arctan(1/|x|)/pi).
//      Then, using g(x) for arctan(1/|x|)/pi:
//        arctan(x)/pi = sign(x) * (1/2 - g(1/|x|)).
//
//      Note that if 1/|x| still falls outside the
//      g(x)'s primary range of accuracy (i.e., if 0.5 < 1/|x| <= 1), the rule
//      from Case 2 must be applied recursively to 1/|x|.

LLVM_LIBC_FUNCTION(float16, atanpif16, (float16 x)) {
  return math::atanpif16(x);
}

} // namespace LIBC_NAMESPACE_DECL
