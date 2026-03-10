//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/opencl/convert.h>

#define __CLC_S_SCALAR_TYPE_DST __CLC_SCALAR_TYPE_DST
#define __CLC_U_SCALAR_TYPE_DST __CLC_XCONCAT(u, __CLC_SCALAR_TYPE_DST)

#define __CLC_S_GENTYPE_DST                                                    \
  __CLC_XCONCAT(__CLC_S_SCALAR_TYPE_DST, __CLC_VECSIZE)
#define __CLC_U_GENTYPE_DST                                                    \
  __CLC_XCONCAT(__CLC_U_SCALAR_TYPE_DST, __CLC_VECSIZE)

#define __CLC_FUNCTION_S __CLC_XCONCAT(convert_, __CLC_S_GENTYPE_DST)
#define __CLC_FUNCTION_U __CLC_XCONCAT(convert_, __CLC_U_GENTYPE_DST)
#define __CLC_FUNCTION_S_SAT __CLC_XCONCAT(__CLC_FUNCTION_S, _sat)
#define __CLC_FUNCTION_U_SAT __CLC_XCONCAT(__CLC_FUNCTION_U, _sat)
#define __CLC_IMPL_FUNCTION_S __CLC_XCONCAT(__clc_convert_, __CLC_S_GENTYPE_DST)
#define __CLC_IMPL_FUNCTION_U __CLC_XCONCAT(__clc_convert_, __CLC_U_GENTYPE_DST)
#define __CLC_IMPL_FUNCTION_S_SAT __CLC_XCONCAT(__CLC_IMPL_FUNCTION_S, _sat)
#define __CLC_IMPL_FUNCTION_U_SAT __CLC_XCONCAT(__CLC_IMPL_FUNCTION_U, _sat)

#define __CLC_F2I

#define __CLC_SCALAR_TYPE_DST char
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_DST

#define __CLC_SCALAR_TYPE_DST short
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_DST

#define __CLC_SCALAR_TYPE_DST int
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_DST

#define __CLC_SCALAR_TYPE_DST long
#define __CLC_BODY <convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_DST
