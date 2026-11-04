//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_remquo.h"

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/integer/clc_clz.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_flush_if_daz.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_frexp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_rint.h"
#include "clc/math/clc_subnormal_config.h"
#include "clc/math/clc_trunc.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isfinite.h"
#include "clc/relational/clc_isnan.h"
#include "clc/relational/clc_signbit.h"

#define __CLC_FUNCTION __clc_remquo_stret
#define __CLC_BODY "clc_remquo_stret.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION

#define __CLC_FUNCTION __clc_remquo
#define __CLC_BODY "clc_remquo.inc"
#include "clc/math/gentype.inc"

#define __CLC_OUT_ARG3_SCALAR_TYPE int
#define __CLC_OUT_ARG3_ADDRESS_SPACE __private
#define __CLC_BODY "clc/shared/binary_with_out_arg_scalarize.inc"
#include "clc/math/gentype.inc"
#undef __CLC_OUT_ARG3_ADDRESS_SPACE

#define __CLC_OUT_ARG3_SCALAR_TYPE int
#define __CLC_OUT_ARG3_ADDRESS_SPACE __local
#define __CLC_BODY "clc/shared/binary_with_out_arg_scalarize.inc"
#include "clc/math/gentype.inc"
#undef __CLC_OUT_ARG3_ADDRESS_SPACE

#define __CLC_OUT_ARG3_SCALAR_TYPE int
#define __CLC_OUT_ARG3_ADDRESS_SPACE __global
#define __CLC_BODY "clc/shared/binary_with_out_arg_scalarize.inc"
#include "clc/math/gentype.inc"
#undef __CLC_OUT_ARG3_ADDRESS_SPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_OUT_ARG3_SCALAR_TYPE int
#define __CLC_OUT_ARG3_ADDRESS_SPACE
#define __CLC_BODY "clc/shared/binary_with_out_arg_scalarize.inc"
#include "clc/math/gentype.inc"
#undef __CLC_OUT_ARG3_ADDRESS_SPACE
#endif
