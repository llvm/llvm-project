//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_GEOMETRIC_FAST_NORMALIZE_H__
#define __CLC_OPENCL_GEOMETRIC_FAST_NORMALIZE_H__

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION fast_normalize
#define __CLC_GEOMETRIC_RET_GENTYPE
#define __CLC_BODY <clc/geometric/unary_decl.inc>

#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION
#undef __CLC_GEOMETRIC_RET_GENTYPE

#endif // __CLC_OPENCL_GEOMETRIC_FAST_NORMALIZE_H__
