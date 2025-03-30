//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifdef __CLC_SCALAR
#define __CLC_FUNCTION __CLC_XCONCAT(__clc_native_, __CLC_NATIVE_INTRINSIC)
#define __CLC_INTRINSIC "llvm." __CLC_XSTR(__CLC_NATIVE_INTRINSIC)

#undef cl_khr_fp64
#include <clc/math/unary_intrin.inc>

#endif

#define __CLC_FUNCTION __CLC_XCONCAT(native_, __CLC_NATIVE_INTRINSIC)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE val) {
  return __CLC_XCONCAT(__clc_native_, __CLC_NATIVE_INTRINSIC)(val);
}

#undef __CLC_FUNCTION
