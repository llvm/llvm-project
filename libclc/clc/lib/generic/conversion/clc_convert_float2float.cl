//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/common/clc_sign.h>
#include <clc/float/definitions.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_nextafter.h>
#include <clc/relational/clc_select.h>
#include <clc/shared/clc_clamp.h>

#define __CLC_GENTYPE_SRC __CLC_XCONCAT(__CLC_SCALAR_TYPE_SRC, __CLC_VECSIZE)

#define __CLC_GENTYPE_SRC_S                                                    \
  __CLC_XCONCAT(__CLC_SCALAR_TYPE_SRC_S, __CLC_VECSIZE)

#define __CLC_FUNCTION __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define __CLC_SCALAR_TYPE_SRC half
#define __CLC_SCALAR_TYPE_SRC_S short
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_SCALAR_TYPE_SRC_S
#endif // cl_khr_fp16

#define __CLC_SCALAR_TYPE_SRC float
#define __CLC_SCALAR_TYPE_SRC_S int
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_SCALAR_TYPE_SRC_S

#ifdef cl_khr_fp64
#define __CLC_SCALAR_TYPE_SRC double
#define __CLC_SCALAR_TYPE_SRC_S long
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_SCALAR_TYPE_SRC_S
#endif // cl_khr_fp64

#undef __CLC_GENTYPE_SRC
#undef __CLC_GENTYPE_SRC_S
#undef __CLC_FUNCTION
