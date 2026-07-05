//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_exp2.h"
#include "clc/math/clc_exp2_fast.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_log.h"
#include "clc/math/clc_log2.h"
#include "clc/math/clc_log2_fast.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_trunc.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_isunordered.h"

#define __CLC_COMPILING_POWR
#define __CLC_BODY "clc_pow_base.inc"
#include "clc/math/gentype.inc"

#define __CLC_FUNCTION __clc_powr
#define __CLC_BODY "clc/shared/binary_def_scalarize.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_powr_fast
#define __CLC_BODY "clc/shared/binary_def_scalarize.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
