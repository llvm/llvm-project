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
#include <clc/integer/clc_abs.h>
#include <clc/integer/definitions.h>
#include <clc/math/clc_nextafter.h>
#include <clc/relational/clc_select.h>
#include <clc/shared/clc_clamp.h>
#include <clc/shared/clc_max.h>
#include <clc/shared/clc_min.h>

#define __CLC_GENTYPE_SRC __CLC_XCONCAT(__CLC_SCALAR_TYPE_SRC, __CLC_VECSIZE)

#define __CLC_GENTYPE_SRC_S                                                    \
  __CLC_XCONCAT(__CLC_SCALAR_TYPE_SRC_S, __CLC_VECSIZE)
#define __CLC_GENTYPE_SRC_U                                                    \
  __CLC_XCONCAT(__CLC_XCONCAT(u, __CLC_SCALAR_TYPE_SRC_S), __CLC_VECSIZE)

#define __CLC_FUNCTION __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)

#define __CLC_I2F

#define __CLC_SCALAR_TYPE_SRC_S char
#define __CLC_GENSIZE_SRC 8

#define __CLC_GEN_S
#define __CLC_SCALAR_TYPE_SRC char
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GEN_S

#define __CLC_SCALAR_TYPE_SRC uchar
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#undef __CLC_SCALAR_TYPE_SRC_S
#undef __CLC_GENSIZE_SRC
#define __CLC_SCALAR_TYPE_SRC_S short
#define __CLC_GENSIZE_SRC 16

#define __CLC_GEN_S
#define __CLC_SCALAR_TYPE_SRC short
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GEN_S

#define __CLC_SCALAR_TYPE_SRC ushort
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#undef __CLC_SCALAR_TYPE_SRC_S
#undef __CLC_GENSIZE_SRC
#define __CLC_SCALAR_TYPE_SRC_S int
#define __CLC_GENSIZE_SRC 32

#define __CLC_GEN_S
#define __CLC_SCALAR_TYPE_SRC int
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GEN_S

#define __CLC_SCALAR_TYPE_SRC uint
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#undef __CLC_SCALAR_TYPE_SRC_S
#undef __CLC_GENSIZE_SRC
#define __CLC_SCALAR_TYPE_SRC_S long
#define __CLC_GENSIZE_SRC 64

#define __CLC_GEN_S
#define __CLC_SCALAR_TYPE_SRC long
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
#undef __CLC_GEN_S

#define __CLC_SCALAR_TYPE_SRC ulong
#define __CLC_BODY <clc_convert_float.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#undef __CLC_GENTYPE_SRC
#undef __CLC_GENTYPE_SRC_S
#undef __CLC_GENTYPE_SRC_U
#undef __CLC_FUNCTION
#undef __CLC_I2F
#undef __CLC_SCALAR_TYPE_SRC_S
#undef __CLC_GENSIZE_SRC
