//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_lgamma_r.h"

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/math/clc_div_fast.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_log.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_sinpi.h"
#include "clc/math/clc_trunc.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_isnan.h"

#define __CLC_FUNCTION __clc_lgamma_r_stret
#define __CLC_BODY "clc_lgamma_r_stret.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION

#define __CLC_FUNCTION __clc_lgamma_r
#define __CLC_BODY "clc_lgamma_r.inc"
#include "clc/math/gentype.inc"

#define __CLC_OUT_ARG2_SCALAR_TYPE int
#define __CLC_ADDRSPACE __private
#define __CLC_BODY "clc/shared/unary_with_out_arg_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_ADDRSPACE

#define __CLC_OUT_ARG2_SCALAR_TYPE int
#define __CLC_ADDRSPACE __global
#define __CLC_BODY "clc/shared/unary_with_out_arg_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_ADDRSPACE

#define __CLC_OUT_ARG2_SCALAR_TYPE int
#define __CLC_ADDRSPACE __local
#define __CLC_BODY "clc/shared/unary_with_out_arg_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_ADDRSPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_OUT_ARG2_SCALAR_TYPE int
#define __CLC_ADDRSPACE __generic
#define __CLC_BODY "clc/shared/unary_with_out_arg_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_ADDRSPACE
#endif
