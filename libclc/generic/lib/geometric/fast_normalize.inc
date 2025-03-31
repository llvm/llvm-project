//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_SCALAR

// Only handle vector implementations
_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN fast_normalize(__CLC_FLOATN p) {
  __CLC_FLOAT l2 = dot(p, p);
  return l2 == 0.0f ? p : p * half_rsqrt(l2);
}

#endif
