//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/opencl/convert.h>

#define __CLC_GENTYPE_SRC __CLC_XCONCAT(__CLC_SCALAR_TYPE_SRC, __CLC_VECSIZE)

#define __CLC_FUNCTION __CLC_XCONCAT(convert_, __CLC_GENTYPE)
#define __CLC_IMPL_FUNCTION __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)

#define __CLC_F2F

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define __CLC_SCALAR_TYPE_SRC half
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#endif // cl_khr_fp16

#define __CLC_SCALAR_TYPE_SRC float
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#ifdef cl_khr_fp64
#define __CLC_SCALAR_TYPE_SRC double
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#endif // cl_khr_fp64
