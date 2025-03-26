//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_SCALAR

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return FUNCTION_IMPL(x, y);
}

#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x, float y) {
  __CLC_GENTYPE vec_y = (__CLC_GENTYPE) (y);
  return FUNCTION_IMPL(x, vec_y);
}

#ifdef cl_khr_fp64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x, double y) {
  __CLC_GENTYPE vec_y = (__CLC_GENTYPE) (y);
  return FUNCTION_IMPL(x, vec_y);
}

#endif
