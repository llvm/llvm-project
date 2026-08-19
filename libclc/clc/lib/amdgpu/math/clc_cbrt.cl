//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_cbrt.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_div_fast.h"
#include "clc/math/clc_exp2_fast.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_flush_if_daz.h"
#include "clc/math/clc_frexp_exp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_log2_fast.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_rint.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isinf.h"

// TODO: This does not require scalarization, but for the moment vectorizing
// results in worse code for f16.

#define __CLC_BODY "clc_amdgpu_cbrt.inc"
#include "clc/math/gentype.inc"

#define __CLC_FUNCTION __clc_cbrt
#define __CLC_BODY "clc/shared/unary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
