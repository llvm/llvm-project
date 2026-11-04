//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_exp2.h"
#include "clc/math/clc_exp2_fast.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_log2.h"
#include "clc/math/clc_log2_fast.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_pown.h"
#include "clc/math/clc_trunc.h"
#include "clc/relational/clc_isinf.h"

#define __CLC_COMPILING_POWN
#define __CLC_BODY "clc_pow_base.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION

#define __CLC_ARG2_SCALAR_TYPE int
#define __CLC_FUNCTION __clc_pown
#define __CLC_BODY "clc/shared/binary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION

#define __CLC_FLOAT_ONLY
#define __CLC_ARG2_SCALAR_TYPE int
#define __CLC_FUNCTION __clc_pown_fast
#define __CLC_BODY "clc/shared/binary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
