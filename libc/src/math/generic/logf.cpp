//===-- Single-precision log(x) function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/logf.h"
#include "src/__support/math/logf.h" // Lookup table for (1/f) and log(f)

// This is an algorithm for log(x) in single precision which is correctly
// rounded for all rounding modes, based on the implementation of log(x) from
// the RLIBM project at:
// https://people.cs.rutgers.edu/~sn349/rlibm

// Step 1 - Range reduction:
//   For x = 2^m * 1.mant, log(x) = m * log(2) + log(1.m)
//   If x is denormal, we normalize it by multiplying x by 2^23 and subtracting
//   m by 23.

// Step 2 - Another range reduction:
//   To compute log(1.mant), let f be the highest 8 bits including the hidden
// bit, and d be the difference (1.mant - f), i.e. the remaining 16 bits of the
// mantissa. Then we have the following approximation formula:
//   log(1.mant) = log(f) + log(1.mant / f)
//               = log(f) + log(1 + d/f)
//               ~ log(f) + P(d/f)
// since d/f is sufficiently small.
//   log(f) and 1/f are then stored in two 2^7 = 128 entries look-up tables.

// Step 3 - Polynomial approximation:
//   To compute P(d/f), we use a single degree-5 polynomial in double precision
// which provides correct rounding for all but few exception values.
//   For more detail about how this polynomial is obtained, please refer to the
// paper:
//   Lim, J. and Nagarakatte, S., "One Polynomial Approximation to Produce
// Correctly Rounded Results of an Elementary Function for Multiple
// Representations and Rounding Modes", Proceedings of the 49th ACM SIGPLAN
// Symposium on Principles of Programming Languages (POPL-2022), Philadelphia,
// USA, January 16-22, 2022.
// https://people.cs.rutgers.edu/~sn349/papers/rlibmall-popl-2022.pdf

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, logf, (float x)) { return math::logf(x); }

} // namespace LIBC_NAMESPACE_DECL
