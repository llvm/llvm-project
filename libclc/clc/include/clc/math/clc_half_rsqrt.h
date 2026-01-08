//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_CLC_HALF_RSQRT_H__
#define __CLC_MATH_CLC_HALF_RSQRT_H__

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_half_rsqrt
#define __CLC_BODY <clc/shared/unary_decl.inc>

#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION

#endif // __CLC_MATH_CLC_HALF_RSQRT_H__
