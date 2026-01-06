//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/integer/definitions.h>
#include <clc/shared/clc_clamp.h>
#include <clc/shared/clc_max.h>
#include <clc/shared/clc_min.h>

#define __CLC_S_SCALAR_TYPE_SRC __CLC_SCALAR_TYPE_SRC
#define __CLC_U_SCALAR_TYPE_SRC __CLC_XCONCAT(u, __CLC_SCALAR_TYPE_SRC)

#define __CLC_S_GENTYPE_SRC                                                    \
  __CLC_XCONCAT(__CLC_S_SCALAR_TYPE_SRC, __CLC_VECSIZE)
#define __CLC_U_GENTYPE_SRC                                                    \
  __CLC_XCONCAT(__CLC_U_SCALAR_TYPE_SRC, __CLC_VECSIZE)

#define __CLC_FUNCTION __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)
#define __CLC_FUNCTION_SAT __CLC_XCONCAT(__CLC_FUNCTION, _sat)

#define __CLC_SCALAR_TYPE_SRC char
#define __CLC_GENSIZE_SRC 8
#define __CLC_BODY <clc_convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GENSIZE_SRC

#define __CLC_SCALAR_TYPE_SRC short
#define __CLC_GENSIZE_SRC 16
#define __CLC_BODY <clc_convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GENSIZE_SRC

#define __CLC_SCALAR_TYPE_SRC int
#define __CLC_GENSIZE_SRC 32
#define __CLC_BODY <clc_convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GENSIZE_SRC

#if defined cles_khr_int64 || !defined(__EMBEDDED_PROFILE__)
#define __CLC_SCALAR_TYPE_SRC long
#define __CLC_GENSIZE_SRC 64
#define __CLC_BODY <clc_convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GENSIZE_SRC
#endif // defined cles_khr_int64 || !defined(__EMBEDDED_PROFILE__)

#undef __CLC_S_SCALAR_TYPE_SRC
#undef __CLC_U_SCALAR_TYPE_SRC
#undef __CLC_S_GENTYPE_SRC
#undef __CLC_U_GENTYPE_SRC
#undef __CLC_FUNCTION
#undef __CLC_FUNCTION_SAT
