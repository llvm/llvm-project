//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/opencl/convert.h>

#define __CLC_S_SCALAR_TYPE_SRC __CLC_SCALAR_TYPE_SRC
#define __CLC_U_SCALAR_TYPE_SRC __CLC_XCONCAT(u, __CLC_SCALAR_TYPE_SRC)

#define __CLC_S_GENTYPE_SRC                                                    \
  __CLC_XCONCAT(__CLC_S_SCALAR_TYPE_SRC, __CLC_VECSIZE)
#define __CLC_U_GENTYPE_SRC                                                    \
  __CLC_XCONCAT(__CLC_U_SCALAR_TYPE_SRC, __CLC_VECSIZE)

#define __CLC_FUNCTION __CLC_XCONCAT(convert_, __CLC_GENTYPE)
#define __CLC_FUNCTION_SAT __CLC_XCONCAT(__CLC_FUNCTION, _sat)
#define __CLC_IMPL_FUNCTION __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)
#define __CLC_IMPL_FUNCTION_SAT __CLC_XCONCAT(__CLC_IMPL_FUNCTION, _sat)

#define __CLC_SCALAR_TYPE_SRC char
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#define __CLC_SCALAR_TYPE_SRC short
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#define __CLC_SCALAR_TYPE_SRC int
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#define __CLC_SCALAR_TYPE_SRC long
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
